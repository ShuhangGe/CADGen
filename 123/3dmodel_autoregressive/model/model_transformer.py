import sys 
sys.path.append("..") 
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
##from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
#from knn_cuda import KNN
##from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from .model_utils import _make_seq_first, _make_batch_first, \
    _get_padding_mask, _get_key_padding_mask, _get_group_mask
    
from .unet3d import ResidualUNet3D
from .resnet_backbone import ResNet50

import logging
import copy

from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
import utils.parser as parser
from .attention import MultiheadAttention
from .transformer import _get_activation_fn
from .build import MODELS
from .text_encoder import Text_Encoder

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
    # 因为前面是num_heads个头一起参与的计算，所以这里要进行一下变形，以便于后面计算。 且同时交换了0，1两个维度
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [batch_size * num_heads,src_len,kdim]
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [batch_size * num_heads,src_len,vdim]
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    # [batch_size * num_heads,tgt_len,kdim] x [batch_size * num_heads, kdim, src_len]
    # =  [batch_size * num_heads, tgt_len, src_len]  这就num_heads个QK相乘后的注意力矩阵

    if attn_mask is not None:
        attn_output_weights += attn_mask  # [batch_size * num_heads, tgt_len, src_len]

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        # 变成 [batch_size, num_heads, tgt_len, src_len]的形状
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'))  #
        # 扩展维度，key_padding_mask从[batch_size,src_len]变成[batch_size,1,1,src_len]
        # 然后再对attn_output_weights进行填充
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len,
                                                       src_len)  # [batch_size * num_heads, tgt_len, src_len]

    attn_output_weights = F.softmax(attn_output_weights, dim=-1)  # [batch_size * num_heads, tgt_len, src_len]
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    # Z = [batch_size * num_heads, tgt_len, src_len]  x  [batch_size * num_heads,src_len,vdim]
    # = # [batch_size * num_heads,tgt_len,vdim]
    # 这就num_heads个Attention(Q,K,V)结果

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    # 先transpose成 [tgt_len, batch_size* num_heads ,kdim]
    # 再view成 [tgt_len,batch_size,num_heads*kdim]
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)

    Z = out_proj(attn_output)
    # 这里就是多个z  线性组合成Z  [tgt_len,batch_size,embed_dim]

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
class FCN(nn.Module):
    def __init__(self, d_model, n_commands, n_args, args_dim=256):
        super().__init__()

        self.n_args = n_args
        self.args_dim = args_dim+ 1

        self.command_fcn = nn.Linear(d_model, n_commands)
        self.args_fcn = nn.Linear(d_model, n_args * self.args_dim)
        

    def forward(self, out):
        S, N, _ = out.shape

        command_logits = self.command_fcn(out)  # Shape [S, N, n_commands]

        args_logits = self.args_fcn(out)  # Shape [S, N, n_args * args_dim]
        args_logits = args_logits.reshape(S, N, self.n_args, self.args_dim)  # Shape [S, N, n_args, args_dim]

        return command_logits, args_logits
@MODELS.register_module()
class Views2Points(nn.Module):
    '''
    input : picture:(batchsize, channel, H, W)
    output: (batchsize, channel, H, W, Z)
    '''
    def __init__(self,config):
        super().__init__()
        self.img_feature = ResNet50(resnet_out = int(config.resnet_out/4))
        self.unet = ResidualUNet3D(config.resnet_out*3,config.UNet_out,num_levels = 4)
        self.config = config
        
        d_model_transdecoder = config.d_model_transdecoder
        nhead_transdecoder = config.nhead_transdecoder
        num_encoder_layers_transdecoder = config.num_encoder_layers_transdecoder
        num_decoder_layers_transdecoder = config.num_decoder_layers_transdecoder
        dim_feedforward_transdecoder = config.dim_feedforward_transdecoder
        dropout_transdecoder = config.dropout_transdecoder
        
        self.text_encoder = Text_Encoder(config)

        decoder_layer = MyTransformerDecoderLayer(d_model_transdecoder, nhead_transdecoder, dim_feedforward_transdecoder, dropout_transdecoder)
        decoder_norm = nn.LayerNorm(d_model_transdecoder)
        self.decoder_trans = MyTransformerDecoder(decoder_layer, num_decoder_layers_transdecoder, decoder_norm)
        self.grid_sample = config.grid_sample
        self.fcn = FCN(self.config.d_model, self.config.n_commands, self.config.n_args, self.config.args_dim)

    def encoder(self, side,front,top,cad_data,command,paramaters):
        side_feature = self.img_feature(side)
        front_feature = self.img_feature(front)
        top_feature = self.img_feature(top)
        print('side_feature.shape: ',side_feature.shape)
        print('front_features.shape: ',front_feature.shape)
        print('top_feature.shape: ',top_feature.shape)
        '''side_feature.shape:  torch.Size([10, 256, 8, 8])
            front_features.shape:  torch.Size([10, 256, 8, 8])
            top_feature.shape:  torch.Size([10, 256, 8, 8])'''
        assert side_feature.shape[-1]==front_feature.shape[-1]==top_feature.shape[-1]==side_feature.shape[-2]==front_feature.shape[-2]==top_feature.shape[-2]
        repeat_num = side_feature.shape[-1]
        side_3d = side_feature.unsqueeze(-3).repeat(1,1,repeat_num,1,1)
        front_3d = front_feature.unsqueeze(-2).repeat(1,1,1,repeat_num,1)
        top_3d = top_feature.unsqueeze(-1).repeat(1,1,1,1,repeat_num)
        print('side_3d.shape: ',side_3d.shape)
        print('front_3d.shape: ',front_3d.shape)
        print('top_3d.shape: ',top_3d.shape)
        '''side_3d.shape:  torch.Size([10, 256, 8, 8, 8])
            front_3d.shape:  torch.Size([10, 256, 8, 8, 8])
            top_3d.shape:  torch.Size([10, 256, 8, 8, 8])'''
        feature_3d = torch.cat((side_3d,front_3d,top_3d),dim=1)
        print('feature_3d1.shape: ',feature_3d.shape)
        '''feature_3d1.shape:  torch.Size([10, 768, 8, 8, 8])'''
        feature_3d = self.unet(feature_3d)
        print('feature_3d2.shape: ',feature_3d.shape)
        '''feature_3d2.shape:  torch.Size([10, 256, 8, 8, 8])'''
        data = feature_3d.float()

        #data = F.grid_sample(feature_3d, cad_data.view(-1,self.grid_sample,self.grid_sample,self.grid_sample,3), mode='bilinear', padding_mode='zeros', align_corners=None)
        
        
        print('data.shape: ',data.shape)
        '''data.shape:  torch.Size([10, 256, 8, 8, 8])'''
        image_feature = data.view(data.shape[0],data.shape[1],-1).permute(0,2,1)
        command = command.clamp(0, 6)
        paramaters = paramaters.clamp(-1, 255)
        text = self.text_encoder(command, paramaters)
        print('text.shape: ',text.shape)
        '''text.shape:  torch.Size([63, 10, 256])'''
        text_feature = text.permute(1,0,2)

        return image_feature, text_feature
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
    def forward(self,side,front,top,cad_data,command,paramaters):
        image_feature, text_feature = self.encoder(side,front,top,cad_data,command,paramaters)
        #print('image_feature.shape: ',image_feature.shape)
        #print('text_feature.shape: ',text_feature.shape)
        '''image_feature.shape:  torch.Size([10, 512, 256])
        text_feature.shape:  torch.Size([10, 63, 256])'''
        # image_feat = F.normalize(torch.max(image_feature, dim=1, keepdim=True)[0])  
        # print('image.shape: ',image.shape)
        # text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)
        # # get momentum features
        # with torch.no_grad():
        #     self._momentum_update()
        #     image_embeds_m = self.visual_encoder_m(image) 
        #     image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
        #     image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)                   
            
        #     text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,                      
        #                                         return_dict = True, mode = 'text')    
        #     text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
        #     text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

        #     sim_i2t_m = image_feat_m @ text_feat_all / self.temp  
        #     sim_t2i_m = text_feat_m @ image_feat_all / self.temp 

        #     sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
        #     sim_targets.fill_diagonal_(1)          

        #     sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
        #     sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        
        # sim_i2t = image_feat @ text_feat_all / self.temp
        # sim_t2i = text_feat @ image_feat_all / self.temp
                             
        # loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        # loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        # loss_ita = (loss_i2t+loss_t2i)/2
        
        image_feature = image_feature.permute(1,0,2)
        text_feature = text_feature.permute(1,0,2)
        #print('image_feature.shape: ',image_feature.shape)
        #print('text_feature.shape: ',text_feature.shape)
        '''image_feature.shape:  torch.Size([512, 10, 256])
        text_feature.shape:  torch.Size([63, 10, 256])'''

        res = self.decoder(image_feature, text_feature)
        return res

        
if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = parser.get_args()
    side = torch.rand(1,3,64,64).to(device)  
    front = torch.rand(1,3,64,64).to(device)
    top = torch.rand(1,3,64,64).to(device)
    cad_data = torch.rand(1, 1024, 3).to(device)
    '''
    side((x),y,z)
    front(x,(y),z)
    top:(x,y,(z))
    '''
    model = Views2Points(cfg=cfg).to(device)
    print(model)
    a= model(side,front,top,cad_data)
        
        