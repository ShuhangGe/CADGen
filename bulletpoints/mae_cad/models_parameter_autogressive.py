# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_1d_sincos_pos_embed_from_grid
import torch.nn.functional as F
import logging
from loss import CADLoss
from model_utils import _make_seq_first, _make_batch_first, \
    _get_padding_mask, _get_key_padding_mask, _get_group_mask
from layers.transformer import *
from layers.improved_transformer import *
from layers.positional_encoding import *


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MyTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(MyTransformerDecoderLayer, self).__init__()
        """
        :param d_model:         d_k = d_v = d_model/nhead = 64, 模型中向量的维度，论文默认值为 512
        :param nhead:           多头注意力机制中多头的数量，论文默认为值 8
        :param dim_feedforward: 全连接中向量的维度，论文默认值为 2048
        :param dropout:         丢弃率，论文中的默认值为 0.1    
        """
        self.self_attn = MyMultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        # 解码部分输入序列之间的多头注意力（也就是论文结构图中的Masked Multi-head attention)
        self.multihead_attn = MyMultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        # 编码部分输出（memory）和解码部分之间的多头注意力机制。
        # Implementation of Feedforward model

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        :param tgt:  解码部分的输入，形状为 [tgt_len,batch_size, embed_dim]
        :param memory: 编码部分的输出（memory）, [src_len,batch_size,embed_dim]
        :param tgt_mask: 注意力Mask输入，用于掩盖当前position之后的信息, [tgt_len, tgt_len]
        :param memory_mask: 编码器-解码器交互时的注意力掩码，一般为None
        :param tgt_key_padding_mask: 解码部分输入的padding情况，形状为 [batch_size, tgt_len]
        :param memory_key_padding_mask: 编码部分输入的padding情况，形状为 [batch_size, src_len]
        :return:
        """
        #print('tgt_mask.shape: ',tgt_mask.shape)
        #print('memory_mask: ',memory_mask)#memory_mask:  None
        #print('tgt_key_padding_mask.shape: ',tgt_key_padding_mask.shape)
        #print('memory_key_padding_mask.shape: ',memory_key_padding_mask.shape)
        '''tgt_mask.shape:  torch.Size([35, 35])
        tgt_key_padding_mask.shape:  torch.Size([128, 35])
        memory_key_padding_mask.shape:  torch.Size([128, 35])'''
        #logging.info(tgt_mask)
        '''tgt_mask:  tensor([[0., -inf, -inf,  ..., -inf, -inf, -inf],
                [0., 0., -inf,  ..., -inf, -inf, -inf],
                [0., 0., 0.,  ..., -inf, -inf, -inf],
                ...,
                [0., 0., 0.,  ..., 0., -inf, -inf],
                [0., 0., 0.,  ..., 0., 0., -inf],
                [0., 0., 0.,  ..., 0., 0., 0.]])
        memory_mask:  None
        tgt_key_padding_mask:  tensor([[False, False, False,  ...,  True,  True,  True],
                [False, False, False,  ...,  True,  True,  True],
                [False, False, False,  ...,  True,  True,  True],
                ...,
                [False, False, False,  ...,  True,  True,  True],
                [False, False, False,  ...,  True,  True,  True],
                [False, False, False,  ...,  True,  True,  True]])
        memory_key_padding_mask:  tensor([[False, False, False,  ..., False, False, False],
                [False, False, False,  ...,  True,  True,  True],
                [False, False, False,  ...,  True,  True,  True],
                ...,
                [False, False, False,  ...,  True,  True,  True],
                [False, False, False,  ...,  True,  True,  True],
                [False, False, False,  ...,  True,  True,  True]])'''
        tgt2 = self.self_attn(tgt, tgt, tgt,  # [tgt_len,batch_size, embed_dim]
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # 解码部分输入序列之间'的多头注意力（也就是论文结构图中的Masked Multi-head attention)

        tgt = tgt + self.dropout1(tgt2)  # 接着是残差连接
        tgt = self.norm1(tgt)  # [tgt_len,batch_size, embed_dim]
        print('tgt.shape: ',tgt.shape)
        print('memory.shape: ',memory.shape)
        tgt2 = self.multihead_attn(tgt, memory, memory,  # [tgt_len, batch_size, embed_dim]
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        # 解码部分的输入经过多头注意力后同编码部分的输出（memory）通过多头注意力机制进行交互
        tgt = tgt + self.dropout2(tgt2)  # 残差连接
        tgt = self.norm2(tgt)  # [tgt_len, batch_size, embed_dim]

        tgt2 = self.activation(self.linear1(tgt))  # [tgt_len, batch_size, dim_feedforward]
        tgt2 = self.linear2(self.dropout(tgt2))  # [tgt_len, batch_size, embed_dim]
        # 最后的两层全连接
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt  # [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]


class MyTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(MyTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        :param tgt: 解码部分的输入，形状为 [tgt_len,batch_size, embed_dim]
        :param memory: 编码部分最后一层的输出 [src_len,batch_size, embed_dim]
        :param tgt_mask: 注意力Mask输入，用于掩盖当前position之后的信息, [tgt_len, tgt_len]
        :param memory_mask: 编码器-解码器交互时的注意力掩码，一般为None
        :param tgt_key_padding_mask: 解码部分输入的padding情况，形状为 [batch_size, tgt_len]
        :param memory_key_padding_mask: 编码部分输入的padding情况，形状为 [batch_size, src_len]
        :return:
        """
        output = tgt  # [tgt_len,batch_size, embed_dim]

        for mod in self.layers:  # 这里的layers就是N层解码层堆叠起来的
            output = mod(output, memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output  # [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]

class MyMultiheadAttention(nn.Module):
    """
    多头注意力机制的计算公式为（就是论文第5页的公式）：
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super(MyMultiheadAttention, self).__init__()
        """
        :param embed_dim:   词嵌入的维度，也就是前面的d_model参数，论文中的默认值为512
        :param num_heads:   多头注意力机制中多头的数量，也就是前面的nhead参数， 论文默认值为 8
        :param dropout:     
        :param bias:        最后对多头的注意力（组合）输出进行线性变换时，是否使用偏置
        """
        self.embed_dim = embed_dim  # 前面的d_model参数
        self.head_dim = embed_dim // num_heads  # head_dim 指的就是d_k,d_v
        self.kdim = self.head_dim
        self.vdim = self.head_dim

        self.num_heads = num_heads  # 多头个数
        self.dropout = dropout

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim 除以 num_heads必须为整数"
        # 上面的限制条件就是论文中的  d_k = d_v = d_model/n_head 条件

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # embed_dim = kdim * num_heads
        # 这里第二个维度之所以是embed_dim，实际上这里是同时初始化了num_heads个W_q堆叠起来的, 也就是num_heads个头
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # W_k,  embed_dim = kdim * num_heads
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # W_v,  embed_dim = vdim * num_heads
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 最后将所有的Z组合起来的时候，也是一次性完成， embed_dim = vdim * num_heads
        self._reset_parameters()

    def _reset_parameters(self):
        """
        以特定方式来初始化参数
        :return:
        """
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        在论文中，编码时query, key, value 都是同一个输入， 解码时 输入的部分也都是同一个输入，
        解码和编码交互时 key,value指的是 memory, query指的是tgt
        :param query: # [tgt_len, batch_size, embed_dim], tgt_len 表示目标序列的长度
        :param key:  #  [src_len, batch_size, embed_dim], src_len 表示源序列的长度
        :param value: # [src_len, batch_size, embed_dim], src_len 表示源序列的长度
        :param attn_mask: # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
                一般只在解码时使用，为了并行一次喂入所有解码部分的输入，所以要用mask来进行掩盖当前时刻之后的位置信息
        :param key_padding_mask: [batch_size, src_len], src_len 表示源序列的长度
        :return:
        attn_output: [tgt_len, batch_size, embed_dim]
        attn_output_weights: # [batch_size, tgt_len, src_len]
        """
        return multi_head_attention_forward(query, key, value, self.num_heads,
                                            self.dropout,
                                            out_proj=self.out_proj,
                                            training=self.training,
                                            key_padding_mask=key_padding_mask,
                                            q_proj=self.q_proj,
                                            k_proj=self.k_proj,
                                            v_proj=self.v_proj,
                                            attn_mask=attn_mask)


def multi_head_attention_forward(query,  # [tgt_len,batch_size, embed_dim]
                                 key,  # [src_len, batch_size, embed_dim]
                                 value,  # [src_len, batch_size, embed_dim]
                                 num_heads,
                                 dropout_p,
                                 out_proj,  # [embed_dim = vdim * num_heads, embed_dim = vdim * num_heads]
                                 training=True,
                                 key_padding_mask=None,  # [batch_size,src_len/tgt_len]
                                 q_proj=None,  # [embed_dim,kdim * num_heads]
                                 k_proj=None,  # [embed_dim, kdim * num_heads]
                                 v_proj=None,  # [embed_dim, vdim * num_heads]
                                 attn_mask=None,  # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
                                 ):
    #print('key_padding_mask:',key_padding_mask)
    #print('query.shape: ',query.shape)
    #print('key.shape: ',key.shape)
    #print('value.shape: ',value.shape)
    q = q_proj(query)
    #  [tgt_len,batch_size, embed_dim] x [embed_dim,kdim * num_heads] = [tgt_len,batch_size,kdim * num_heads]

    k = k_proj(key)
    # [src_len, batch_size, embed_dim] x [embed_dim, kdim * num_heads] = [src_len, batch_size, kdim * num_heads]

    v = v_proj(value)
    # [src_len, batch_size, embed_dim] x [embed_dim, vdim * num_heads] = [src_len, batch_size, vdim * num_heads]

    tgt_len, bsz, embed_dim = query.size()  # [tgt_len,batch_size, embed_dim]
    src_len = key.size(0)
    head_dim = embed_dim // num_heads  # num_heads * head_dim = embed_dim
    scaling = float(head_dim) ** -0.5
    q = q * scaling  # [query_len,batch_size,kdim * num_heads]

    if attn_mask is not None:  # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)  # [1, tgt_len,src_len]
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        # 现在 atten_mask 的维度就变成了3D

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    # [batch_size * num_heads,tgt_len,kdim]
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [batch_size * num_heads,src_len,kdim]
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [batch_size * num_heads,src_len,vdim]
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    # [batch_size * num_heads,tgt_len,kdim] x [batch_size * num_heads, kdim, src_len]
    # =  [batch_size * num_heads, tgt_len, src_len]  这就num_heads个QK相乘后的注意力矩阵

    if attn_mask is not None:
        attn_output_weights += attn_mask  # [batch_size * num_heads, tgt_len, src_len]

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'))  #
        # key_padding_mask: [batch_size,src_len]->[batch_size,1,1,src_len]
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len,
                                                       src_len)  # [batch_size * num_heads, tgt_len, src_len]

    attn_output_weights = F.softmax(attn_output_weights, dim=-1)  # [batch_size * num_heads, tgt_len, src_len]
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    # Z = [batch_size * num_heads, tgt_len, src_len]  x  [batch_size * num_heads,src_len,vdim]
    # = # [batch_size * num_heads,tgt_len,vdim]

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    # [tgt_len, batch_size* num_heads ,kdim]
    # [tgt_len,batch_size,num_heads*kdim]
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)

    Z = out_proj(attn_output)

    return Z, attn_output_weights.sum(dim=1) / num_heads  # average attention weights over heads

def _generate_future_mask(
         size: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        # Default mask is for forward direction. Flip for backward direction.
        mask = torch.triu(
            torch.ones(size, size, device=device, dtype=dtype), diagonal=1
        )
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

class MyTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(MyTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        output = tgt  # [tgt_len,batch_size, embed_dim]

        for mod in self.layers:  
            output = mod(output, memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output  # [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]




class CADEmbedding(nn.Module):
    """Embedding: positional embed + command embed + parameter embed + group embed (optional)"""
    def __init__(self, args, seq_len, use_group=False):
        super().__init__()
        self.command_embed = nn.Embedding(args.n_commands, args.embed_dim)
        args_dim = args.args_dim + 1
        self.arg_embed = nn.Embedding(args_dim, 64, padding_idx=0)
        self.embed_fcn = nn.Linear(64 * args.n_args, args.embed_dim)

    def forward(self, commands, args, groups=None):
        S, N = commands.shape
        command_embed = self.command_embed(commands.long())
        src =  command_embed+ \
              self.embed_fcn(self.arg_embed((args + 1).long()).view(S, N, -1))  # shift due to -1 PAD_VAL
        # src = self.pos_encoding(src)
        return src,command_embed
    
class FCN(nn.Module):
    def __init__(self, d_model, n_commands, n_args, args_dim=256):
        super().__init__()

        self.n_args = n_args
        self.args_dim = args_dim+ 1
        self.d_model = d_model
        # self.command_fcn = nn.Linear(d_model, n_commands)
        self.args_fcn = nn.Linear(d_model, n_args * self.args_dim)
        

    def forward(self, out):
        S, N, _ = out.shape
        # print('out.shape: ',out.shape)
        # print('self.d_model: ',self.d_model)
        # command_logits = self.command_fcn(out)  # Shape [S, N, n_commands]

        args_logits = self.args_fcn(out)  # Shape [S, N, n_args * args_dim]
        args_logits = args_logits.reshape(S, N, self.n_args, self.args_dim)  # Shape [S, N, n_args, args_dim]

        return args_logits

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, args, mask_ratio,
                 embed_dim=256, depth=24, num_heads=16,
                 decoder_embed_dim=128, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics

        self.mask_ratio = mask_ratio
        self.max_len = args.max_total_len
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_len, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        if args.device == 'cpu':
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.embedding = CADEmbedding(args, args.max_total_len)
        
        encoder_layer = TransformerEncoderLayerImproved(args.embed_dim, args.num_heads, args.dim_feedforward, args.dropout)
        encoder_norm = LayerNorm(args.embed_dim)
        self.encoder = TransformerEncoder(encoder_layer, args.depth, encoder_norm)
        # --------------------------------------------------------------------------
        # # MAE decoder specifics
        # self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        # self.decoder_command = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.max_len, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])

        # self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, args.n_commands, bias=True) # decoder to patch
        # --------------------------------------------------------------------------
        # autogressive decoder
        d_model_transdecoder = args.d_model_transdecoder
        nhead_transdecoder = args.nhead_transdecoder
        num_encoder_layers_transdecoder = args.num_encoder_layers_transdecoder
        num_decoder_layers_transdecoder = args.num_decoder_layers_transdecoder
        dim_feedforward_transdecoder = args.dim_feedforward_transdecoder
        dropout_transdecoder = args.dropout_transdecoder
        self.cad_pro = nn.Linear(3,args.d_model*3)
        self.text_encoder = Text_Encoder(args)

        decoder_layer = MyTransformerDecoderLayer(d_model_transdecoder, nhead_transdecoder, dim_feedforward_transdecoder, dropout_transdecoder)
        decoder_norm = nn.LayerNorm(d_model_transdecoder)
        self.decoder_trans = MyTransformerDecoder(decoder_layer, num_decoder_layers_transdecoder, decoder_norm)
        self.grid_sample = args.grid_sample
        self.fcn = FCN(self.args.d_model, self.args.n_commands, self.args.n_args, self.args.args_dim)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
        # self.text_encoder = Text_Encoder(args)
        self.args = args.n_commands
        self.embed_dim =embed_dim

        self.fcn = FCN(decoder_embed_dim, args.n_commands, args.n_args)
        
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], int(self.max_len))#########################################
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_pos_embed.shape[-1], int(self.max_len))######################################
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, commmand_embed, padding_mask, key_padding_mask, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_masked = ids_shuffle[:, len_keep:]
        command_masked = torch.gather(commmand_embed, dim=1, index=ids_masked.unsqueeze(-1).repeat(1, 1, D))
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        key_padding_mask_keep = torch.gather(key_padding_mask.unsqueeze(-1), dim=1, index=ids_keep.unsqueeze(-1))
        padding_mask_keep = torch.gather(padding_mask.unsqueeze(-1), dim=1, index=ids_keep.unsqueeze(-1))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_keep, mask, ids_restore, command_masked, padding_mask_keep, key_padding_mask_keep

    def forward_encoder(self, commmand, paramaters, mask_ratio):
        # # embed patches
        '''commmand.shape:  torch.Size([10, 64])
            paramaters.shape:  torch.Size([10, 64, 16])'''
        padding_mask, key_padding_mask = _get_padding_mask(commmand, seq_dim=-1), _get_key_padding_mask(commmand, seq_dim=-1)
        # print('padding_mask.shape: ',padding_mask.shape)
        # print('key_padding_mask.shape: ',key_padding_mask.shape)
        '''
        padding_mask.shape:  torch.Size([10, 64])
            key_padding_mask.shape:  torch.Size([10, 64])
        padding_mask.shape:  torch.Size([64, 10, 1])
            key_padding_mask.shape:  torch.Size([10, 64])'''
        #padding_indicate = torch.sum(padding_mask,dim=-1)
        x, commmand_embed= self.embedding(commmand, paramaters)
        '''text.shape:  torch.Size([20, 64, 256])'''
        x = x + self.pos_embed[:, :, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, command_masked, padding_mask_keep, key_padding_mask_keep= self.random_masking(x, commmand_embed, padding_mask, key_padding_mask, mask_ratio)
        # print('x.shape: ',x.shape)
        padding_mask_keep = padding_mask_keep.permute(1,0,2)
        key_padding_mask_keep = key_padding_mask_keep.squeeze(-1)
        # apply Transformer blocks
        # for blk in self.blocks:
        #     x = blk(x)
        # x = self.norm(x)
        # print('key_padding_mask_keep.shape: ',key_padding_mask_keep.shape)
        x = x.permute(1,0,2)
        x = self.encoder(x, mask=None, src_key_padding_mask=key_padding_mask_keep)
        # print('x0.shape: ',x.shape)
        '''x0.shape:  torch.Size([48(sequence length), 10(batch size), 256])'''
        # print('padding_mask_keep.shape: ',padding_mask_keep.shape)
        x = self.norm(x)
        x = x.permute(1,0,2)
        # print('x1.shape: ',x.shape)
        # print('x: ',x)
        return x, mask, ids_restore,command_masked

    def forward_decoder(self, x, ids_restore,commmand_embed):
        # embed tokens
        print('x.shape: ',x.shape)
        '''torch.Size([256, 48, 256])'''
        x = self.decoder_embed(x)
        temp = torch.zeros(1, 1, x.shape[-1]).to(self.device)
        command_mask  = temp.repeat(x.shape[0], x.shape[1], 1)
        commmand_embed = self.decoder_command(commmand_embed)
        '''commmand_embed.shape:  torch.Size([512, 16, 128])'''
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)#+1 include cls token 
        '''mask_tokens.shape:  torch.Size([512, 16, 128])'''
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        '''x_0.shape:  torch.Size([512, 64, 128])'''
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        '''x_0.shape:  torch.Size([512, 64, 128])'''

        commmand_embed = torch.cat([command_mask, commmand_embed], dim=1)
        commmand_embed = torch.gather(commmand_embed, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        # add pos embed
        x = x + self.decoder_pos_embed
        x = x + commmand_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # print('x1.shape: ',x.shape)
        # predictor projection
        args_logits = self.fcn(x)
        #print('args_logits.shape: ',args_logits.shape)
        #torch.Size([128, 64, 16, 257])
        res = { 
            "args_logits": args_logits
        }        
        return res
   def decoder(self, image_feature, text_feature):
        #attention mask
        future_mask = _generate_future_mask(text_feature.shape[0],text_feature.dtype,text_feature.device)
        #print('future_mask.shape: ',future_mask.shape)
        output = self.decoder_trans(tgt=text_feature, memory=image_feature, tgt_mask=future_mask, memory_mask=None,
                        tgt_key_padding_mask=None,
                        memory_key_padding_mask=None)
        print('output.shape: ',output.shape)
        '''output.shape:  torch.Size([63, 10, 256])'''
        output = output.permute(1,0,2)
        #out_logits = _make_batch_first(*output)
        #print('out_logits.shape: ',out_logits.shape)
        command_logits, args_logits = self.fcn(output)
        print('command_logits.shape: ',command_logits.shape)
        print('args_logits.shape: ',args_logits.shape)
        res = { 
            "command_logits":command_logits,
            "args_logits": args_logits
        }
        #print('out_logits[0].shape: ',out_logits[0].shape)
        '''out_logits[0].shape:  torch.Size([50, 60, 6])'''
        #print('out_logits[1].shape: ',out_logits[1].shape)
        '''out_logits[1].shape:  torch.Size([50, 60, 16, 257])'''
        return res


    def forward(self, command, paramaters, mask_ratio=0.75):
        mask_ratio = self.mask_ratio
        # text = self.text_encoder(command, paramaters,self.pos_embed)
        #expand one dimension to command
        latent, mask, ids_restore, command_masked= self.forward_encoder(command, paramaters, mask_ratio)
        #print('latent.shape: ',latent.shape)
        #print('mask.shape: ',mask.shape)
        #print('ids_restore.shape: ',ids_restore.shape)
        '''latent.shape:  torch.Size([20, 16, 256])
            mask.shape:  torch.Size([20, 64])
            ids_restore.shape:  torch.Size([20, 64])'''
        pred = self.forward_decoder(latent, ids_restore, command_masked)  # [N, L, p*p*3]
        # loss_dict = loss_fun(output)
        return pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
