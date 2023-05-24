import sys 
sys.path.append("..") 
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
#from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from knn_cuda import KNN
#from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from .model_utils import _make_seq_first, _make_batch_first, \
    _get_padding_mask, _get_key_padding_mask, _get_group_mask

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


class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


# class TransformerDecoder(nn.Module):
#     def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate,
#                 drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
#             )
#             for i in range(depth)])
#         self.norm = norm_layer(embed_dim)
#         self.head = nn.Identity()

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def forward(self, x, pos, return_token_num):
#         for _, block in enumerate(self.blocks):
#             x = block(x + pos)

#         x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
#         return x
# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.mask_ratio 
        self.trans_dim = config.trans_dim
        self.depth = config.depth 
        self.drop_path_rate = config.drop_path_rate
        self.num_heads = config.num_heads 

        # embedding
        self.encoder_dims =  config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        self.mask_type = config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, neighborhood, center, noaug = False):
        # generate mask
        # if self.mask_type == 'rand':
        #     bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        # else:
        #     bool_masked_pos = self._mask_center_block(center, noaug = noaug)

        group_input_tokens = self.encoder(neighborhood)  #  B G C

        # add pos embedding
        # mask pos center

        pos = self.pos_embed(center)

        # transformer
        x_vis = self.blocks(group_input_tokens, pos)
        x_vis = self.norm(x_vis)
        #print()

        return x_vis
# decoder from deepcad
'''
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, memory2=None, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt
        #print(output)
        #print('output.shape: ',output.shape)
        '''output.shape:  torch.Size([60, 10, 256])'''
        #print('self.layers: ',self.layers )
        for mod in self.layers:
            #print('mod: ',mod)
            output = mod(output, memory, memory2=memory2, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
class TransformerDecoderLayerGlobalImproved(Module):
    #cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout
    def __init__(self, d_model, d_global, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", d_global2=None):
        super(TransformerDecoderLayerGlobalImproved, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear_global = Linear(d_global, d_model)

        if d_global2 is not None:
            self.linear_global2 = Linear(d_global2, d_model)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout2_2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayerGlobalImproved, self).__setstate__(state)

    def forward(self, tgt, memory, memory2=None, tgt_mask=None, tgt_key_padding_mask=None, *args, **kwargs):

        '''tgt.shape:  torch.Size([60, 50, 256])'''
        tgt1 = self.norm1(tgt)
        '''tgt1.shape:  torch.Size([60, 50, 256])'''
        tgt2 = self.self_attn(tgt1, tgt1, tgt1, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        '''tgt2.shape:  torch.Size([60, 50, 256])'''
        tgt = tgt + self.dropout1(tgt2)
        '''tgt.shape:  torch.Size([60, 50, 256])'''
        tgt2 = self.linear_global(memory)
        '''tgt2.shape:  torch.Size([1, 50, 256])'''
        tgt = tgt + self.dropout2(tgt2)  # implicit broadcast
        '''tgt.shape:  torch.Size([60, 50, 256])'''
        
        if memory2 is not None:
            tgt2_2 = self.linear_global2(memory2)
            tgt = tgt + self.dropout2_2(tgt2_2)

        tgt1 = self.norm2(tgt)
        '''tgt1.shape:  torch.Size([60, 50, 256])'''
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt1))))
        '''tgt2.shape:  torch.Size([60, 50, 256])'''
        tgt = tgt + self.dropout3(tgt2)
        '''tgt.shape:  torch.Size([60, 50, 256])'''
        return tgt
class PositionalEncodingLUT(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncodingLUT, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer('position', position)

        self.pos_embed = nn.Embedding(max_len, d_model)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x):
        #print('x.shape: ',x.shape)
        '''x.shape:  torch.Size([60, 50, 256])'''
        
        pos = self.position[:x.size(0)]
        #print('pos: ',pos)
        # print('pos.shape: ',pos.shape)
        # print('self.pos_embed(pos).shape: ',self.pos_embed(pos).shape)
        '''pos.shape:  torch.Size([60, 1])
        self.pos_embed(pos).shape:  torch.Size([60, 1, 256])'''
        x = x + self.pos_embed(pos)
        #print('x.shape: ',x.shape)
        #x.shape:  torch.Size([60, 50, 256])
        return self.dropout(x)
    
class ConstEmbedding(nn.Module):
    """learned constant embedding"""
    def __init__(self, cfg, seq_len):
        super().__init__()

        self.d_model = cfg.d_model
        self.seq_len = seq_len

        self.PE = PositionalEncodingLUT(cfg.d_model, max_len=seq_len)

    def forward(self, z):
        N = z.size(1)
        
        # np.set_printoptions(threshold=np.inf)
        #print('z.new_zeros(self.seq_len, N, self.d_model).shape:',z.new_zeros(self.seq_len, N, self.d_model).shape)
        '''z.new_zeros(self.seq_len, N, self.d_model).shape: torch.Size([60, 50, 256])'''
        #print('z.new_zeros(self.seq_len, N, self.d_model): ',z.new_zeros(self.seq_len, N, self.d_model)[1,1,:].detach().cpu().numpy())
        '''all zeros'''
        src = self.PE(z.new_zeros(self.seq_len, N, self.d_model))
        return src
class FCN(nn.Module):
    def __init__(self, d_model, n_commands, n_args, args_dim=256):
        super().__init__()

        self.n_args = n_args
        self.args_dim = args_dim

        self.command_fcn = nn.Linear(d_model, n_commands)
        self.args_fcn = nn.Linear(d_model, n_args * args_dim)

    def forward(self, out):
        S, N, _ = out.shape

        command_logits = self.command_fcn(out)  # Shape [S, N, n_commands]

        args_logits = self.args_fcn(out)  # Shape [S, N, n_args * args_dim]
        args_logits = args_logits.reshape(S, N, self.n_args, self.args_dim)  # Shape [S, N, n_args, args_dim]

        return command_logits, args_logits
class Bottleneck(nn.Module):
    def __init__(self, cfg):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Sequential(nn.Linear(cfg.d_model, cfg.dim_z),
                                        nn.Tanh())

    def forward(self, z):
        return self.bottleneck(z)

class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()

        self.embedding = ConstEmbedding(cfg, cfg.max_total_len)

        decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)

        args_dim = cfg.args_dim + 1
        self.fcn = FCN(cfg.d_model, cfg.n_commands, cfg.n_args, args_dim)
        self.linear = nn.Sequential(
            nn.Linear(cfg.num_group*cfg.encoder_dims,cfg.dim_z),
        )


        self.bottleneck = Bottleneck(cfg)
    def forward(self, z):
        
        '''linear'''
        # z = z.view(z.shape[0],-1).unsqueeze(0)
        # z = self.linear(z)
        '''max'''
        z = torch.max(z, dim=1, keepdim=True)[0]
        z = z.permute(1,0,2)
        z = self.bottleneck(z)
        src = self.embedding(z)
        #print('src.shape: ',src.shape)
        out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)
        #print('out.shape: ',out.shape)
        command_logits, args_logits = self.fcn(out)

        out_logits = (command_logits, args_logits)
        return out_logits
# new decoder
'''
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
'''


class UNet(nn.Module):
    def __init__(self, in_channel=96, out_channel=2, training=True):
        super(UNet, self).__init__()
        self.training = training
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv3d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4=   nn.Conv3d(128, 256, 3, stride=1, padding=1)
        # self.encoder5=   nn.Conv3d(256, 512, 3, stride=1, padding=1)
        
        # self.decoder1 = nn.Conv3d(512, 256, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder2 =   nn.Conv3d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 =   nn.Conv3d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv3d(32, 2, 3, stride=1, padding=1)
        
        self.map4 = nn.Sequential(
            nn.Conv3d(2, out_channel, 1, 1),
            #nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Softmax(dim =1)
        )
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(16, 32, 32), mode='trilinear'),
            nn.Softmax(dim =1)
        )
        self.padding = torch.nn.ReplicationPad3d((1, 0, 1, 0, 1, 0))

    def forward(self, x):

        out = F.relu(F.max_pool3d(self.encoder1(x),2,2))
        t1 = out
        out = F.relu(F.max_pool3d(self.encoder2(out),2,2))
        t2 = out
        
        out = F.relu(F.max_pool3d(self.encoder3(out),2,2))
        t3 = out
        out = F.relu(F.max_pool3d(self.encoder4(out),2,2))
        #output1 = self.map1(out)
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t3)
        #output2 = self.map2(out)
        #print('out1.shape: ',out.shape)
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2,2),mode ='trilinear'))
        #print('out2.shape: ',out.shape)
        #print('t2.shape: ',t2.shape)
        if out.shape[-1] !=t2.shape[-1]:
            out = self.padding(out)
            print('out3.shape: ',out.shape)
        out = torch.add(out,t2)
        #output3 = self.map3(out)
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t1)
        #print('out1.shape: ',out.shape)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2,2),mode ='trilinear'))
        #print('out2.shape: ',out.shape)
        output4 = self.map4(out)
        #print('output4.shape: ',output4.shape)
        return output4

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResnetBlock, self).__init__()
        self.conv0 = nn.Conv2d(3, in_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,  # downsample with first conv
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = F.relu(x)
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition

        return y
@MODELS.register_module()
class Views2Points(nn.Module):
    '''
    input : picture:(batchsize, channel, H, W)
    output: (batchsize, channel, H, W, Z)
    '''
    def __init__(self,config):
        super().__init__()
        self.img_feature =ResnetBlock(config.resnet_in,config.resnet_out,2)
        self.conv3d = nn.Conv3d(in_channels=96,
                        out_channels=96,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        padding=0)
        self.unet = UNet(in_channel=config.resnet_out*3,out_channel=config.UNet_out)
        self.config = config
        self.trans_dim = config.trans_dim #384
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size#32
        self.num_group = config.num_group#64
        self.drop_path_rate = config.drop_path_rate#0.1
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.decoder_depth
        self.decoder_num_heads = config.decoder_num_heads
        
        self.MAE_decoder = Decoder(config)

        #print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_MAE')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        self.grid_sample = config.grid_sample
        self.text_encoder = Text_Encoder(config)

    def forward(self, side, front, top, cad_data, command, args):
        # print('model start ')
        # print('side.shape: ',side.shape)
        # print('front.shape: ',front.shape)
        # print('top.shape: ',top.shape)
        # print('cad_data.shape: ',cad_data.shape)
        side_feature = self.img_feature(side)
        front_feature = self.img_feature(front)
        top_feature = self.img_feature(top)
        print('side_feature.shape: ',side_feature.shape)
        #print('front_features.shape: ',front_feature.shape)
        #print('top_feature.shape: ',top_feature.shape)
        assert side_feature.shape[-1]==front_feature.shape[-1]==top_feature.shape[-1]==side_feature.shape[-2]==front_feature.shape[-2]==top_feature.shape[-2]
        repeat_num = side_feature.shape[-1]
        side_3d = side_feature.unsqueeze(-3).repeat(1,1,repeat_num,1,1)
        front_3d = front_feature.unsqueeze(-2).repeat(1,1,1,repeat_num,1)
        top_3d = top_feature.unsqueeze(-1).repeat(1,1,1,1,repeat_num)
        #print('side_3d.shape: ',side_3d.shape)
        #print('front_3d.shape: ',front_3d.shape)
        #print('top_3d.shape: ',top_3d.shape)
        feature_3d = torch.concat((side_3d,front_3d,top_3d),dim=1)
        #print('feature_3d1.shape: ',feature_3d.shape)
        feature_3d = self.unet(feature_3d)
        #print('feature_3d2.shape: ',feature_3d.shape)
        '''side_feature.shape:  torch.Size([1, 32, 64, 64])
        feature_3d1.shape:  torch.Size([1, 96, 64, 64, 64])
        feature_3d2.shape:  torch.Size([1, 32, 64, 64, 64])'''
        '''cad_data.shape:  torch.Size([50, 1024, 3])'''


        data = F.grid_sample(feature_3d, cad_data.view(-1,self.grid_sample,self.grid_sample,self.grid_sample,3), mode='bilinear', padding_mode='zeros', align_corners=None)
        #print('data0.shape: ',data.shape)
        data = data.view(data.shape[0],data.shape[1],-1).permute(0,2,1)
        #print('data.shape: ',data.shape)
                #print('pts.shap: ',pts.shape)
        #print('\n','forwardforwardforwardforward','\n')
        neighborhood, center = self.group_divider(cad_data)
        
        #print('neighborhood.shape:',neighborhood.shape)
        '''neighborhood.shape: torch.Size([50, 32, 32, 3])'''
        #print('center.shape:',center.shape)
        '''center.shape: torch.Size([50, 32, 3])'''
        neighborhood = data.view(data.shape[0],self.num_group,self.group_size,3)
        x_full= self.MAE_encoder(neighborhood, center)
        
        #print('x_full.shape: ',x_full.shape)
        '''x_full.shape:  torch.Size([50, 32, 128])'''
        B,_,C = x_full.shape # B VIS C

        pos_full = self.decoder_pos_embed(center).reshape(B, -1, C)
        #print('pos_full.shape: ',pos_full.shape)
        '''pos_full.shape:  torch.Size([50, 32, 128])'''
        z = x_full+pos_full
        text = self.text_encoder(command, args)
        output = self.MAE_decoder(z)
        out_logits = _make_batch_first(*output)
        res = { 
            "command_logits": out_logits[0],
            "args_logits": out_logits[1]
        }
        #print('out_logits[0].shape: ',out_logits[0].shape)
        '''out_logits[0].shape:  torch.Size([50, 60, 6])'''
        #print('out_logits[1].shape: ',out_logits[1].shape)
        '''out_logits[1].shape:  torch.Size([50, 60, 16, 257])'''
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
        
        