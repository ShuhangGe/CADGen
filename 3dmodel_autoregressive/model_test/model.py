import sys 
sys.path.append("..") 
# from .attention import MultiheadAttention
# from .transformer import _get_activation_fn
# from .build import MODELS
# from .text_encoder import Text_Encoder
# #from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
# from .model_utils import _make_seq_first, _make_batch_first, \
#     _get_padding_mask, _get_key_padding_mask, _get_group_mask

from attention import MultiheadAttention
from transformer import _get_activation_fn
from build import MODELS
from text_encoder import Text_Encoder
#from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from model_utils import _make_seq_first, _make_batch_first, \
    _get_padding_mask, _get_key_padding_mask, _get_group_mask
from bert import BertConfig
from bert.modeling_bert import BertEncoder
        
sys.path.append("/scratch/sg7484/CMDGen/3dmodel_autoregressive") 
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
from PIL import Image
import torchvision.transforms as transforms

#from bert.modeling_bert improt BertEncoder

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
def create_decoder(decoder_type, norm_type,
                   textual_feature_size,
                   attention_heads,
                   feedforward_size,
                   dropout,
                   num_layers,
                   output_hidden_states=False,
                   use_mlp_wrapper=None,
                   ):
    assert norm_type in ['post', 'pre']
    if decoder_type is None:
        assert NotImplemented
    elif decoder_type == 'bert_en':
        from .bert import BertConfig
        from .bert.modeling_bert import BertEncoder
        config = BertConfig(
            vocab_size_or_config_json_file=30522,
            hidden_size=textual_feature_size,
            num_hidden_layers=num_layers,
            num_attention_heads=attention_heads,
            intermediate_size=feedforward_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
        )
        config.pre_norm=(norm_type == 'pre')
        config.use_mlp_wrapper = use_mlp_wrapper
        config.output_hidden_states = output_hidden_states
        encoder = BertEncoder(config)
        return BertEncoderAsDecoder(encoder)

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

class Decoder(nn.Module):
    def __init__(self,cfg):
        super(Decoder,self).__init__()

        
        config = BertConfig(
            vocab_size_or_config_json_file=cfg.args_dim, #256,
            hidden_size=cfg.bert_hidden_size,
            num_hidden_layers=cfg.bert_num_layers,
            num_attention_heads=cfg.bert_attention_heads,
            intermediate_size=cfg.bert_feedforward_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
        )
        norm_type = "post"
        use_mlp_wrapper=None
        output_hidden_states=None
        config.pre_norm=(norm_type == 'pre')
        config.use_mlp_wrapper = use_mlp_wrapper
        config.output_hidden_states = output_hidden_states
        self.decorde  = BertEncoder(config)
        self.fcn = FCN(cfg.d_model, cfg.n_commands, cfg.n_args, cfg.args_dim)
    def forward(self,tgt,memory,
                tgt_mask=None,
                #memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                tgt_bi_valid_mask=None,
                encoder_history_states=None,):
        '''tgt: features from text
            memory; features frome images'''
        assert tgt_key_padding_mask is None, 'not supported'
        assert tgt_mask.dim() == 2
        assert tgt_mask.shape[0] == tgt_mask.shape[1]
        # tgt_mask should always be 0/negative infinity
        # mask
        #print('tgt_mask.shape: ',tgt_mask.shape)
        '''tgt_mask.shape:  torch.Size([13, 13])'''
        #print('tgt_key_padding_mask.shape: ',tgt_key_padding_mask.shape)NONE
        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)
        # print('tgt.shape: ',tgt.shape)
        # print('memory.shape: ',memory.shape)
        '''tgt.shape:  torch.Size([2, 13, 768])
            memory.shape:  torch.Size([2, 101, 768])'''
        hidden_states = torch.cat((memory, tgt), dim=1)
        # print('hidden_states.shape: ',hidden_states.shape)
        '''hidden_states.shape:  torch.Size([2, 114, 768])'''
        num_tgt = tgt.shape[1]
        num_memory = memory.shape[1]
        device = tgt.device
        dtype = tgt.dtype
        top_left = torch.zeros((num_memory, num_memory), device=device, dtype=dtype)
        #print('top_left.shape: ',top_left.shape)
        top_right = torch.full((num_memory, num_tgt), float('-inf'), device=tgt.device, dtype=dtype,)
        #print('top_right.shape: ',top_right.shape)
        bottom_left = torch.zeros((num_tgt, num_memory), dtype=dtype, device=tgt_mask.device,)
        #print('bottom_left.shape: ',bottom_left.shape)
        left = torch.cat((top_left, bottom_left), dim=0)
        right = torch.cat((top_right, tgt_mask.to(dtype)), dim=0)
        #print('left.shape: ',left.shape)
        #print('right.shape: ',right.shape)

        full_attention_mask = torch.cat((left, right), dim=1)[None, :]
        #print('full_attention_mask.shape: ',full_attention_mask.shape)
        '''top_left.shape:  torch.Size([101, 101])
            top_right.shape:  torch.Size([101, 13])
            bottom_left.shape:  torch.Size([13, 101])
            left.shape:  torch.Size([114, 101])
            right.shape:  torch.Size([114, 13])
            full_attention_mask.shape:  torch.Size([1, 114, 114])'''
        #print(memory_key_padding_mask) # None
        if memory_key_padding_mask is None:
            memory_key_padding_mask = torch.full((memory.shape[0], memory.shape[1]), fill_value=False, device=device)
        # if it is False, it means valid. That is, it is not a padding
        assert memory_key_padding_mask.dtype == torch.bool
        zero_negative_infinity = torch.zeros_like(memory_key_padding_mask, dtype=tgt.dtype)
        #print('zero_negative_infinity: ',zero_negative_infinity)
        zero_negative_infinity[memory_key_padding_mask] = float('-inf')
        #print('zero_negative_infinity2: ',zero_negative_infinity)
        #print(zero_negative_infinity)
        full_attention_mask = full_attention_mask.expand((memory_key_padding_mask.shape[0], num_memory + num_tgt, num_memory + num_tgt))
        full_attention_mask = full_attention_mask.clone()
        #print(full_attention_mask)
        origin_left = full_attention_mask[:, :, :num_memory]
        update = zero_negative_infinity[:, None, :]
        full_attention_mask[:, :, :num_memory] = origin_left + update

        if tgt_bi_valid_mask is not None:
            # verify the correctness
            bs = full_attention_mask.shape[0]
            # during inference, tgt_bi_valid_mask's length is not changed, but
            # num_tgt can be increased
            max_valid_target = tgt_bi_valid_mask.shape[1]
            mask = tgt_bi_valid_mask[:, None, :].expand((bs, num_memory+num_tgt, max_valid_target))
            full_attention_mask[:, :, num_memory:(num_memory+max_valid_target)][mask] = 0

        # add axis for multi-head
        full_attention_mask = full_attention_mask[:, None, :, :]
        
        result = self.decorde(
                hidden_states=hidden_states,
                attention_mask=full_attention_mask,
                encoder_history_states=None,
            )
        result = list(result)
        result[0] = result[0][:, num_memory:].transpose(0, 1)
        command_logits, args_logits = self.fcn(result[0])
        #print('command_logits: ',command_logits)
        #print('args_logits: ',args_logits)
        out_logits = (command_logits, args_logits)
        return out_logits
# new decoder
'''
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
'''


class UNet(nn.Module):
    def __init__(self, in_channel=96, out_channel=2, training=True):
        super(UNet, self).__init__()
        self.in_channel = in_channel
        self.training = training
        self.encoder1 =  nn.Sequential(nn.Conv3d(in_channel, 64, 3, stride=1, padding=1),nn.Conv3d(64, 32, 3, stride=1, padding=1))  # b, 16, 10, 10
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

        # self.map3 = nn.Sequential(
        #     nn.Conv3d(64, out_channel, 1, 1),
        #     nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
        #     nn.Softmax(dim =1)
        # )
        # self.map2 = nn.Sequential(
        #     nn.Conv3d(128, out_channel, 1, 1),
        #     nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
        #     nn.Softmax(dim =1)
        # )

        # self.map1 = nn.Sequential(
        #     nn.Conv3d(256, out_channel, 1, 1),
        #     nn.Upsample(scale_factor=(16, 32, 32), mode='trilinear'),
        #     nn.Softmax(dim =1)
        # )
        self.padding = torch.nn.ReplicationPad3d((1, 0, 1, 0, 1, 0))
        self.Tanh = nn.Tanh()
    def forward(self, x):
        #print('self.in_channel: ',self.in_channel)
        #print('x: ', x.shape)
        out = self.encoder1(x)
        #print('out1.shape: ',out.shape)
        out = F.relu(F.max_pool3d(out,2,2))
        t1 = out
        #print('t1.shape: ',t1.shape)
        out = F.relu(F.max_pool3d(self.encoder2(out),2,2))
        t2 = out
        #print('t2.shape: ',t2.shape)
        out = F.relu(F.max_pool3d(self.encoder3(out),2,2))
        t3 = out
        #print('t3.shape: ',t3.shape)
        out = F.relu(F.max_pool3d(self.encoder4(out),2,2))
        #print('out2.shape: ',out.shape)
        #output1 = self.map1(out)
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2,2),mode ='trilinear'))
        #print('out3.shape: ',out.shape)
        out = torch.add(out,t3)
        #output2 = self.map2(out)

        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2,2),mode ='trilinear'))
        #print('out4.shape: ',out.shape)
        if out.shape[-1] !=t2.shape[-1]:
            out = self.padding(out)
            #print('out5.shape: ',out.shape)
        out = torch.add(out,t2)
        #output3 = self.map3(out)
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2,2),mode ='trilinear'))
        #print('out6.shape: ',out.shape)
        out = torch.add(out,t1)

        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2,2),mode ='trilinear'))
        #print('out7.shape: ',out.shape)

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
        if torch.isnan(x).any():
            print('x nan')
        print('x.shape: ',x.shape)
        x = self.conv0(x)
        #print(x)
        if torch.isnan(x).any():
            print('x1 nan')
        x = self.bn0(x)
        if torch.isnan(x).any():
            print('x2 nan')
        x = F.relu(x)
        if torch.isnan(x).any():
            print('x3 nan')
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if torch.isnan(y).any():
            print('y1 nan')
        y = self.bn2(self.conv2(y))
        if torch.isnan(y).any():
            print('y2 nan')
        y += self.shortcut(x)
        if torch.isnan(y).any():
            print('y3 nan')
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        if torch.isnan(y).any():
            print('y4 nan')

        return y
def _generate_future_mask(
         size: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        # Default mask is for forward direction. Flip for backward direction.
        mask = torch.triu(
            torch.ones(size, size, device=device, dtype=dtype), diagonal=1
        )
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask
@MODELS.register_module()
class Views2Points(nn.Module):
    '''
    input : picture:(batchsize, channel, H, W)
    output: (batchsize, channel, H, W, Z)
    '''
    def __init__(self,config):
        super().__init__()
        self.img_feature =ResnetBlock(config.resnet_in,config.resnet_out,2)
        # self.conv3d = nn.Conv3d(in_channels=96,
        #                 out_channels=96,
        #                 kernel_size=(1, 1, 1),
        #                 stride=(1, 1, 1),
        #                 padding=0)
        self.unet = UNet(config.resnet_out*3,config.UNet_out)
        self.config = config
        self.trans_dim = config.trans_dim #384
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size#32
        self.num_group = config.num_group#64
        self.drop_path_rate = config.drop_path_rate#0.1
        #self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        # self.decoder_depth = config.decoder_depth
        # self.decoder_num_heads = config.decoder_num_heads
        
        self.bert_decoder = Decoder(config)

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
        if torch.isnan(side).any():
            print('side nan')
        if torch.isnan(front).any():
            print('front nan')
        if torch.isnan(top).any():
            print('top nan')
        side_feature = self.img_feature(side)
        if torch.isnan(side_feature).any():
            print('img_feature1 nan')
        front_feature = self.img_feature(front)
        if torch.isnan(side_feature).any():
            print('img_feature2 nan') 
        top_feature = self.img_feature(top)
        if torch.isnan(side_feature).any():
            print('img_feature3 nan') 

        #print('side_feature.shape: ',side_feature.shape)
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
        feature_3d = torch.cat((side_3d,front_3d,top_3d),dim=1)
        ##print('feature_3d1.shape: ',feature_3d.shape)
        feature_3d = self.unet(feature_3d)
        if torch.isnan(feature_3d).any():
            print('feature_3d_unet nan')
        #print('feature_3d2.shape: ',feature_3d.shape)
        '''side_feature.shape:  torch.Size([1, 32, 64, 64])
        feature_3d1.shape:  torch.Size([1, 96, 64, 64, 64])
        feature_3d2.shape:  torch.Size([1, 32, 64, 64, 64])'''
        '''cad_data.shape:  torch.Size([50, 1024, 3])'''

        #print('cad_data.shape',cad_data.shape)
        data = F.grid_sample(feature_3d, cad_data.view(-1,self.grid_sample,self.grid_sample,self.grid_sample,3), mode='bilinear', padding_mode='zeros', align_corners=None)
        #print('data1.shape: ',data.shape)
        if torch.isnan(data).any():
            print('data1 nan')
        #print('data0.shape: ',data.shape)
        data = data.view(data.shape[0],data.shape[1],-1).permute(0,2,1)
        print('data2.shape: ',data.shape)
        '''data1.shape:  torch.Size([32, 3, 12, 12, 12])
        data2.shape:  torch.Size([32, 1728, 3])'''
        #print('data.shape: ',data.shape)
                #print('pts.shap: ',pts.shape)
        #print('\n','forwardforwardforwardforward','\n')
        neighborhood, center = self.group_divider(cad_data)
        if torch.isnan(neighborhood).any() or torch.isnan(center).any():
            print('data1 nan')
        
        #print('neighborhood.shape:',neighborhood.shape)
        '''neighborhood.shape: torch.Size([50, 32, 32, 3])[batchsize]'''
        #print('center.shape:',center.shape)
        '''center.shape: torch.Size([50, 32, 3])'''
        neighborhood = data.view(data.shape[0],self.num_group,self.group_size,3)
        x_full= self.MAE_encoder(neighborhood, center)
        if torch.isnan(x_full).any():
            print('x_full nan')
        
        #print('x_full.shape: ',x_full.shape)
        '''x_full.shape:  torch.Size([50, 32, 128])'''
        B,_,C = x_full.shape # B VIS C

        pos_full = self.decoder_pos_embed(center).reshape(B, -1, C)
        if torch.isnan(pos_full).any():
            print('pos_full nan')
        #print('pos_full.shape: ',pos_full.shape)
        '''pos_full.shape:  torch.Size([50, 32, 128])'''
        z = x_full+pos_full
        text = self.text_encoder(command, args)
        if torch.isnan(text).any():
            print('text nan')
        # print('z.shape: ',z.shape)
        # print('text.shape: ',text.shape)
        '''z.shape:  torch.Size([50, 64, 256])[batchsize, num_group, dim]
            text.shape:  torch.Size([64, 50, 256])[tgt_len ,batchsize, dim]'''
        text = text.permute(1,0,2)
        #print('text.shape: ',text.shape)
        #print('z.shape: ',z.shape)
        future_mask = _generate_future_mask(text.shape[0],text.dtype,text.device)
        if torch.isnan(future_mask).any():
            print('future_mask nan')
        out_logits = self.bert_decoder(text,z,future_mask)
        if torch.isnan(out_logits[0]).any() or torch.isnan(out_logits[1]).any():
            print('out_logits nan')
        #print('out_logits[0].shape: ',out_logits[0].shape)
        #print('out_logits[1].shape: ',out_logits[1].shape)
        #output.shape:  torch.Size([50, 64, 256])
        #out_logits = _make_batch_first(*output)
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
    transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([200, 200])
        ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = parser.get_args()
    # side = torch.full((1,3,200,200),255).type(torch.cuda.FloatTensor).to(device)  
    # front = torch.full((1,3,200,200),255).type(torch.cuda.FloatTensor).to(device)
    # top = torch.full((1,3,200,200),255).type(torch.cuda.FloatTensor).to(device)
    front = Image.open('/scratch/sg7484/data/CMDGen/data3D_pic/abc_0000_step_v00/00005885/00005885_f.png')
    side = Image.open('/scratch/sg7484/data/CMDGen/data3D_pic/abc_0000_step_v00/00005885/00005885_r.png')  
    top = Image.open('/scratch/sg7484/data/CMDGen/data3D_pic/abc_0000_step_v00/00005885/00005885_t.png')
    
    front = transforms(front) 
    side = transforms(side) 
    top = transforms(top) 
    front = front.unsqueeze(0).type(torch.cuda.FloatTensor).to(device)  
    side = side.unsqueeze(0).type(torch.cuda.FloatTensor).to(device)  
    top = top.unsqueeze(0).type(torch.cuda.FloatTensor).to(device)  
    print('front.shape: ',front.shape)
    cad_data = torch.rand(1, 1728, 3).type(torch.cuda.FloatTensor).to(device)
    command, args = torch.rand(1, 64).type(torch.cuda.FloatTensor).to(device), torch.rand(1, 64, 16).type(torch.cuda.FloatTensor).to(device)
    '''
    side((x),y,z)
    front(x,(y),z)
    top:(x,y,(z))
    '''
    model = Views2Points(cfg).to(device)
    print(model)
    a= model(side,front,top,cad_data,command, args)
        
        