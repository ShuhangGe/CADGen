import sys 
sys.path.append("..") 
from .attention import MultiheadAttention
from .transformer import _get_activation_fn
from .build import MODELS
from .text_encoder import Text_Encoder
#from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from .model_utils import _make_seq_first, _make_batch_first, \
    _get_padding_mask, _get_key_padding_mask, _get_group_mask
from .unet3d import ResidualUNet3D
from .resnet_backbone import ResNet50
#from .models_ae import AutoEncoder


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
torch.set_printoptions(profile="full")
import logging
from logging import handlers
from model.model_utils import _get_padding_mask, _get_visibility_mask

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))

        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(th)
logname = '/scratch/sg7484/CMDGen/results/paramaters' + '/auto.log'
log = Logger(logname,level='debug')
#log.logger.info(f'-->name: {name}, -->grad_requirs: {param.requires_grad},-->grad_value: {param.grad},\n')
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

    def forward(self, xyz, data_3d):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        #print('xyz.shape: ',xyz.shape)
        #print('data_3d.shape: ',data_3d.shape)
        '''xyz.shape:  torch.Size([12, 1728, 3])'''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group, data_3d) # B G 3
        #print('center.shape: ',center.shape)
        '''center.shape:  torch.Size([12, 64, 3])'''
        # knn to get the neighborhood
        _, idx = self.knn(data_3d, center) # B G M
        #print('idx.shape: ',idx.shape)
        '''idx.shape:  torch.Size([12, 64, 27])'''
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=data_3d.device).view(-1, 1, 1) * num_points
        #print('idx_base.shape: ',idx_base.shape)
        '''idx_base.shape:  torch.Size([12, 1, 1])'''
        idx = idx + idx_base
        idx = idx.view(-1)
        #print('idx.shape: ',idx.shape)
        '''idx.shape:  torch.Size([20736])'''
        neighborhood = data_3d.contiguous().view(batch_size * num_points, -1)[idx, :]
        #print('neighborhood.shape: ',neighborhood.shape)
        '''neighborhood.shape:  torch.Size([20736, 3])'''
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        #print('neighborhood2.shape: ',neighborhood.shape)
        '''neighborhood2.shape:  torch.Size([12, 64, 27, 3])'''

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

        #self.mask_type = config.mask_type

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
        from .bert import BertConfig
        from .bert.modeling_bert import BertEncoder
        
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
            memory; features frome images batch_size =12'''
        assert tgt_key_padding_mask is None, 'not supported'
        assert tgt_mask.dim() == 2
        assert tgt_mask.shape[0] == tgt_mask.shape[1]
        #print('tgt_mask: ',tgt_mask)
        #print('tgt_mask: ',tgt_mask)
        # tgt_mask should always be 0/negative infinity
        # mask
        #print('tgt_mask.shape: ',tgt_mask.shape)
        '''gt_mask.shape:  torch.Size([64, 64])'''
        #print('tgt_key_padding_mask.shape: ',tgt_key_padding_mask.shape)NONE
        # tgt = tgt.transpose(0, 1)
        # memory = memory.transpose(0, 1)
        #print('tgt.shape: ',tgt.shape)
        #print('memory.shape: ',memory.shape)
        '''tgt.shape:  torch.Size([12, 64, 256])
            memory.shape:  torch.Size([12, 64, 256])'''
        hidden_states = torch.cat((memory, tgt), dim=1)
        #print('hidden_states.shape: ',hidden_states.shape)
        '''hidden_states.shape:  torch.Size([12, 114, 768])'''
        num_tgt = tgt.shape[1]
        num_memory = memory.shape[1]
        device = tgt.device
        dtype = tgt.dtype
        top_left = torch.zeros((num_memory, num_memory), device=device, dtype=dtype)
        #print('top_left.shape: ',top_left.shape)
        '''top_left.shape:  torch.Size([64, 64])'''
        top_right = torch.full((num_memory, num_tgt), float('-inf'), device=tgt.device, dtype=dtype,)
        #print('top_right.shape: ',top_right.shape)
        '''top_right.shape:  torch.Size([64, 64])'''
        bottom_left = torch.zeros((num_tgt, num_memory), dtype=dtype, device=tgt_mask.device,)
        #print('bottom_left.shape: ',bottom_left.shape)
        '''bottom_left.shape:  torch.Size([64, 64])'''
        left = torch.cat((top_left, bottom_left), dim=0)
        right = torch.cat((top_right, tgt_mask.to(dtype)), dim=0)
        #print('left.shape: ',left.shape)
        #print('right.shape: ',right.shape)
        '''left.shape:  torch.Size([128, 64])
        right.shape:  torch.Size([128, 64])'''
        full_attention_mask = torch.cat((left, right), dim=1)[None, :]
        #print('full_attention_mask.shape: ',full_attention_mask.shape)
        '''full_attention_mask.shape:  torch.Size([1, 128, 128])'''
        #torch.set_printoptions(threshold=np.inf)
        #print('full_attention_mask: ',full_attention_mask)
        if memory_key_padding_mask is None:
            memory_key_padding_mask = torch.full((memory.shape[0], memory.shape[1]), fill_value=False, device=device)
        #print('memory_key_padding_mask.shape: ',memory_key_padding_mask.shape)
        '''memory_key_padding_mask.shape:  torch.Size([12, 64])'''
        # if it is False, it means valid. That is, it is not a padding
        assert memory_key_padding_mask.dtype == torch.bool
        zero_negative_infinity = torch.zeros_like(memory_key_padding_mask, dtype=tgt.dtype)
        #print('zero_negative_infinity.shape: ',zero_negative_infinity.shape)
        '''zero_negative_infinity.shape:  torch.Size([12, 64])'''
        #print('zero_negative_infinity: ',zero_negative_infinity)
        zero_negative_infinity[memory_key_padding_mask] = float('-inf')
        full_attention_mask = full_attention_mask.expand((memory_key_padding_mask.shape[0], num_memory + num_tgt, num_memory + num_tgt))
        #print('full_attention_mask.shape: ',full_attention_mask.shape)
        '''full_attention_mask.shape:  torch.Size([12, 128, 128])'''
        full_attention_mask = full_attention_mask.clone()
        #print(full_attention_mask)
        origin_left = full_attention_mask[:, :, :num_memory]
        #print('origin_left.shape: ',origin_left.shape)
        '''origin_left.shape:  torch.Size([12, 128, 64])'''
        update = zero_negative_infinity[:, None, :]
        #print('update.shape: ',update.shape)
        '''update.shape:  torch.Size([12, 1, 64])'''
        temp = origin_left + update
        full_attention_mask[:, :, :num_memory] = temp
        #print('temp.shape: ',temp.shape)
        #print('full_attention_mask.shape: ',full_attention_mask.shape)
        '''temp.shape:  torch.Size([12, 128, 64])
        full_attention_mask.shape:  torch.Size([12, 128, 128])'''


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
        #print('full_attention_mask.shape: ',full_attention_mask.shape)
        '''full_attention_mask.shape:  torch.Size([12, 1, 128, 128])'''
        result = self.decorde(
                hidden_states=hidden_states,
                attention_mask=full_attention_mask,
                encoder_history_states=None,
            )
        result = list(result)
        #print('num_memory: ',num_memory)
        #print('num_tgt: ',num_tgt)
        result[0] = result[0][:, num_memory:]#.transpose(0, 1)
        #print('result[0]: ',result[0].shape)
        '''result[0]:  torch.Size([1, 64, 256])'''
        command_logits, args_logits = self.fcn(result[0])
        #print('command_logits.shape: ',command_logits.shape)
        #print('args_logits.shape: ',args_logits.shape)
        '''command_logits.shape:  torch.Size([12, 64, 6])
        args_logits.shape:  torch.Size([12, 64, 16, 257])'''
        out_logits = (command_logits, args_logits)
        return out_logits
# new decoder
'''
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
'''


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
        self.img_feature = ResNet50(resnet_out = int(config.resnet_out/4))

        self.unet = ResidualUNet3D(config.resnet_out*3,config.UNet_out,num_levels = 3)
        self.config = config
        self.trans_dim = config.trans_dim #384
        
        # self.MAE_encoder = MaskTransformer(config)
        # self.group_size = config.group_size#32
        # self.num_group = config.num_group#64
        # self.drop_path_rate = config.drop_path_rate#0.1
        # self.decoder_pos_embed = nn.Sequential(
        #     nn.Linear(3, 128),
        #     nn.GELU(),
        #     nn.Linear(128, self.trans_dim)
        # )
        # #print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_MAE')
        # self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        self.grid_sample = config.grid_sample
        
        self.text_encoder = Text_Encoder(config)
        self.bert_decoder = Decoder(config)
        
    def forward_encoder(self, side, front, top, cad_data, command, args):
        autoregressive_index = command.size(1)
        log.logger.info(f'-->autoregressive_index: {autoregressive_index} \n')
        print('\n \n model start ')
        print('side.shape: ',side.shape)
        print('front.shape: ',front.shape)
        print('top.shape: ',top.shape)
        print('cad_data.shape: ',cad_data.shape)
        print('command.shape: ',command.shape)
        print('args.shape: ',args.shape)

        side_feature = self.img_feature(side)
        front_feature = self.img_feature(front)
        top_feature = self.img_feature(top)
        print('side_feature.shape: ',side_feature.shape)
        print('front_features.shape: ',front_feature.shape)
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
        feature_3d = feature_3d.float()
        print('feature_3d.shape: ',feature_3d.shape)

        data = F.grid_sample(feature_3d, cad_data.view(-1,self.grid_sample,self.grid_sample,self.grid_sample,3), mode='bilinear', padding_mode='zeros', align_corners=None)
        print('data1.shape: ',data.shape)
        data = data.view(data.shape[0],data.shape[1],-1).permute(0,2,1)
        #print('data2.shape: ',data.shape)
        
        # #use mae encoder
        # neighborhood, center = self.group_divider(cad_data,data)
        # #print('neighborhood.shape:',neighborhood.shape)
        # #print('center.shape:',center.shape)
        # '''neighborhood.shape: torch.Size([1, 64, 27, 3])
        # center.shape: torch.Size([1, 64, 3])'''
        # neighborhood = data.view(data.shape[0],self.num_group,self.group_size,3)
        # x_full= self.MAE_encoder(neighborhood, center)
        # #print('x_full[0]: ',x_full[0])
        # #print('x_full.shape: ',x_full.shape)
        # '''x_full.shape:  torch.Size([1, 64, 256])'''
        # B,_,C = x_full.shape # B VIS C
        # pos_full = self.decoder_pos_embed(center).reshape(B, -1, C)
        # #print('pos_full.shape: ',pos_full.shape)
        # '''pos_full.shape:  torch.Size([1, 64, 256])'''
        # z = x_full+pos_full
        # #z = self.attention_encode(data)
        
        return data
    def forward_decoder(self,z,command, args):
        command = command.clamp(0, 6)
        args = args.clamp(-1, 255)
        text = self.text_encoder(command, args)
        #print('z.shape: ',z.shape)
        #print('text.shape: ',text.shape)
        '''z.shape:  torch.Size([1, 64, 256])
        text.shape:  torch.Size([64, 1, 256])'''
        text = text.permute(1,0,2)
        #print('text.shape: ',text.shape)
        '''text.shape:  torch.Size([1, 64, 256])'''
        future_mask = _generate_future_mask(text.shape[1],text.dtype,text.device)
        #print('future_mask: ',future_mask)
        #print('future_mask.shape: ',future_mask.shape)
        '''future_mask.shape:  torch.Size([64, 64])'''
        #print('text.shape: ',text.shape)
        # print('text[0]: ',text[0])
        # print('z[0]: ',z[0])
        out_logits = self.bert_decoder(text,z,future_mask)

        #out_logits = _make_batch_first(*output)
        res = { 
            "command_logits": out_logits[0],
            "args_logits": out_logits[1]
        }
        #print('out_logits[0].shape: ',out_logits[0].shape)
        #print('out_logits[1].shape: ',out_logits[1].shape)
        #print('\n')
        '''out_logits[0].shape:  torch.Size([1, 64, 6])
        out_logits[1].shape:  torch.Size([1, 64, 16, 257])'''
        return res
    def forward(self, side, front, top, cad_data, command, args):
        
        z = self.forward_encoder(side, front, top, cad_data, command, args)
        res = self.forward_decoder(z,command, args)
        return res
        
if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = parser.get_args()
    side = torch.full((1,3,200,200),255).to(device)  
    front = torch.full((1,3,200,200),255).to(device)
    top = torch.full((1,3,200,200),255).to(device)
    cad_data = torch.rand(1, 1024, 3).to(device)
    '''
    side((x),y,z)
    front(x,(y),z)
    top:(x,y,(z))
    '''
    model = Views2Points(cfg=cfg).to(device)
    print(model)
    a= model(side,front,top,cad_data)
        
        