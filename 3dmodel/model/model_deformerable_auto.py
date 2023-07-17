import sys 
sys.path.append("..") 
import torch
from torch import nn, Tensor

import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
#from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
#from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from .model_utils import _make_seq_first, _make_batch_first, \
    _get_padding_mask, _get_key_padding_mask, _get_group_mask
    
from .unet3d import ResidualUNet3D
from .resnet_backbone import ResNet50
from typing import Optional, List

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
from .unet3d import ResidualUNet3D
from utils.check_paramaters import check_para
from .deformable_attn_3d import DeformableHeadAttention
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

def _get_clones(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])
class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 k: int,
                 scales: int,
                 last_point_num: int,

                 need_attn: bool = False,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ms_deformbale_attn = DeformableHeadAttention(h=nhead,
                                                          d_model=d_model,
                                                          k=k,
                                                          scales=scales,
                                                          last_point_num=last_point_num,
                                                          dropout=dropout,
                                                          need_attn=need_attn)

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

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.need_attn = need_attn
        self.attns = []

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     ref_point: Tensor,
                     tgt_mask: Optional[Tensor] = None,
                     memory_masks: Optional[List[Tensor]] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_masks: Optional[List[Tensor]] = None,
                     poses: Optional[List[Tensor]] = None,
                     query_pos: Optional[Tensor] = None):
        #print('\ndecoder start')
        # #check_para(tgt = tgt, memory = memory, ref_point = ref_point, tgt_mask=tgt_mask, memory_masks=memory_masks,tgt_key_padding_mask=tgt_key_padding_mask,\
        #     memory_key_padding_masks=memory_key_padding_masks,poses=poses,query_pos=query_pos)

        '''all zeros'''
        #print('query_pos: ',query_pos)
        '''position embedding form nn.Embedding(num_queries, hidden_dim)'''
        
        q = k = self.with_pos_embed(tgt, query_pos)
        #print('q.shape: ',q.shape)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        #memory = [self.with_pos_embed(tensor, pos) for tensor, pos in zip(memory, poses)]

        # L, B, C -> B, L, 1, C
        tgt = tgt.transpose(0, 1).unsqueeze(dim=2)
        ref_point = ref_point.transpose(0, 1).unsqueeze(dim=2)

        # B, L, 1, C
        tgt2, attns = self.ms_deformbale_attn(tgt,
                                              memory,
                                              ref_point,
                                              query_mask=None,
                                              key_masks=memory_key_padding_masks)
        #check_para('model_deformerable',tgt2 = tgt2)

        if self.need_attn:
            self.attns.append(attns)

        tgt = tgt + self.dropout2(tgt2)
        #check_para('model_deformerable',tgt2 = tgt2)
        tgt = self.norm2(tgt)
        #check_para('model_deformerable',tgt2 = tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        #check_para('model_deformerable',tgt2 = tgt2)
        tgt = tgt + self.dropout3(tgt2)
        #check_para('model_deformerable',tgt2 = tgt2)
        tgt = self.norm3(tgt)
        #check_para('model_deformerable',tgt2 = tgt2)

        # B, L, 1, C -> L, B, C
        tgt = tgt.squeeze(dim=2)
        #check_para('model_deformerable',tgt2 = tgt2)
        tgt = tgt.transpose(0, 1).contiguous()
        #check_para('model_deformerable',tgt2 = tgt2)

        # decoder we only one query tensor
        #print('decoder end\n')
        return tgt

    def forward_pre(self, tgt: Tensor,
                    memory: List[Tensor],
                    ref_point: Tensor,
                    tgt_mask: Optional[Tensor] = None,
                    memory_masks: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_masks: Optional[Tensor] = None,
                    poses: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):

        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)

        #memory = [self.with_pos_embed(tensor, pos) for tensor, pos in zip(memory, poses)]

        # L, B, C -> B, L, 1, C
        tgt2 = tgt2.transpose(0, 1).unsqueeze(dim=2)
        ref_point = ref_point.transpose(0, 1).unsqueeze(dim=2)

        # B, L, 1, 2
        tgt2, attns = self.ms_deformbale_attn(tgt2, memory, ref_point,
                                              query_mask=None,
                                              key_masks=memory_key_padding_masks)
        if self.need_attn:
            self.attns.append(attns)

        # B, L, 1, C -> L, B, C
        tgt2 = tgt2.squeeze(dim=2)
        tgt2 = tgt2.transpose(0, 1).contiguous()

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

    def forward(self, tgt: Tensor,
                memory: List[Tensor],
                ref_point: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_masks: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_masks: Optional[Tensor] = None,
                poses: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, ref_point, tgt_mask, memory_masks,
                                    tgt_key_padding_mask, memory_key_padding_masks, poses, query_pos)
        return self.forward_post(tgt, memory, ref_point, tgt_mask, memory_masks,
                                 tgt_key_padding_mask, memory_key_padding_masks, poses, query_pos)
        
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                ref_point: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_masks: Optional[List[Tensor]] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_masks: Optional[List[Tensor]] = None,
                poses: Optional[List[Tensor]] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            #print('one layer')
            output = layer(output,
                           memory,
                           ref_point,
                           tgt_mask=tgt_mask,
                           memory_masks=memory_masks,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_masks=memory_key_padding_masks,
                           poses=poses, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

# new decoder
'''
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
class FCN(nn.Module):
    def __init__(self, d_model, n_commands, n_args, args_dim=256):
        super().__init__()

        self.n_args = n_args
        self.args_dim = args_dim

        self.command_fcn = nn.Linear(d_model, n_commands)
        self.args_fcn = nn.Linear(d_model, n_args * args_dim)

    def forward(self, out):
        
        if len(out.size())==4:
            S, N, _,dim = out.shape
            out = out.view(S, N,dim)
        else:
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
        self.grid_sample = config.grid_sample
        
        #deocder
        hidden_dim = config.d_model_deformable
        self.query_embed = nn.Embedding(config.num_queries, hidden_dim)
        self.query_embed  = self.query_embed.weight
        self.query_ref_point_proj = nn.Linear(config.d_model_deformable, 3)
        decoder_layer = TransformerDecoderLayer(d_model=config.d_model_deformable,
                                        nhead=config.nhead_deformable,
                                        k=config.k_deformabel,
                                        scales=config.scales_deformable,
                                        last_point_num=config.last_point_num,
                                        
                                        dim_feedforward=config.dim_feedforward_deformable,
                                        dropout=0.1,
                                        activation="relu",
                                        normalize_before=False)
        decoder_norm = nn.LayerNorm(config.d_model_deformable)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers_deformable, norm = decoder_norm,
                                          return_intermediate=False)

        args_dim = config.args_dim + 1
        self.fcn = FCN(config.d_model, config.n_commands, config.n_args, args_dim)
        #print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_MAE')

    def forward(self,side,front,top,cad_data):
        # print('model start ')
        # print('side.shape: ',side.shape)
        # print('front.shape: ',front.shape)
        # print('top.shape: ',top.shape)
        # print('cad_data.shape: ',cad_data.shape)
        side_feature = self.img_feature(side)
        front_feature = self.img_feature(front)
        top_feature = self.img_feature(top)
        #print('side_feature.shape: ',side_feature.shape)
        #print('front_features.shape: ',front_feature.shape)
        print('top_feature.shape: ',top_feature.shape)
        assert side_feature.shape[-1]==front_feature.shape[-1]==top_feature.shape[-1]==side_feature.shape[-2]==front_feature.shape[-2]==top_feature.shape[-2]
        repeat_num = side_feature.shape[-1]
        side_3d = side_feature.unsqueeze(-3).repeat(1,1,repeat_num,1,1)
        front_3d = front_feature.unsqueeze(-2).repeat(1,1,1,repeat_num,1)
        top_3d = top_feature.unsqueeze(-1).repeat(1,1,1,1,repeat_num)
        print('side_3d.shape: ',side_3d.shape)
        print('front_3d.shape: ',front_3d.shape)
        print('top_3d.shape: ',top_3d.shape)
        feature_3d = torch.cat((side_3d,front_3d,top_3d),dim=1)
        print('feature_3d1.shape: ',feature_3d.shape)
        feature_3d = self.unet(feature_3d)
        print('feature_3d2.shape: ',feature_3d.shape)
        
        feature_3d = feature_3d.float()
        feature_3d = feature_3d.permute(0,2,3,4,1)
        print('feature_3d2.shape: ',feature_3d.shape)
        decoder_features = [feature_3d for _ in range(self.config.scales_deformable)]
        check_para('model_deformerable',decoder_features = decoder_features)
        # data = F.grid_sample(feature_3d, cad_data.view(-1,self.grid_sample,self.grid_sample,self.grid_sample,-1), mode='bilinear', padding_mode='zeros', align_corners=None)
        # print('data0.shape: ',data.shape)
        #data = data.view(data.shape[0],data.shape[1],-1).permute(0,2,1)
        bs = feature_3d.size(0)
        query_embed = self.query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        query_ref_point = self.query_ref_point_proj(tgt)
        query_ref_point = F.sigmoid(query_ref_point)
        output = self.decoder(tgt,decoder_features,query_ref_point,query_pos=query_embed)
        #check_para('model_deformerable',output=output)
        command_logits, args_logits = self.fcn(output)
        output = (command_logits, args_logits)
        out_logits = _make_batch_first(*output)
        #check_para('model_deformerable',out_logits0=out_logits[0])
        #check_para('model_deformerable',out_logits1=out_logits[1])
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
        
        