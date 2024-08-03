'''pair with main_deocder, generate cmd from a guassian distribution, and input command'''
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
import numpy as np
import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_1d_sincos_pos_embed_from_grid
import torch.nn.functional as F
from loss import CADLoss
from model_utils import _make_seq_first, _make_batch_first, \
    _get_padding_mask, _get_key_padding_mask, _get_group_mask
from layers.transformer import *
from layers.improved_transformer import *
from layers.positional_encoding import *
import logging
import sys
np.set_printoptions(suppress=True)

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
        if args.device == 'cpu':
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = args.max_total_len
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_command = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.max_len, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, args.n_commands, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
        # self.text_encoder = Text_Encoder(args)
        self.args = args.n_commands
        self.embed_dim =embed_dim

        self.fcn = FCN(decoder_embed_dim, args.n_commands, args.n_args)
        self.command_embed = nn.Embedding(args.n_commands, args.embed_dim)
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

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


    def forward_decoder(self, commands, x):
        '''(decoder_embed): Linear(in_features=256, out_features=128, bias=True)
            torch.Size([256, 48, 256])
            torch.Size([256, 64, 256])'''
        command_embed = self.command_embed(commands.long())
        x = self.decoder_embed(x)
        command_embed = self.decoder_command(command_embed)
        # add pos embed
        x = x + self.decoder_pos_embed
        x = x + command_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        args_logits = self.fcn(x)

        '''args_logits:  torch.Size([256, 64, 16, 257])'''
        # print('commands1.shape',commands.shape)
        paramaters = F.softmax(args_logits,dim=-1)
        paramaters = torch.argmax(paramaters,dim=-1)
        commands = commands.unsqueeze(-1)
        # print('paramaters.shape: ',paramaters.shape)
        # print('commands2.shape: ',commands.shape)
        '''commands1.shape torch.Size([256, 64])
        paramaters.shape:  torch.Size([256, 64, 16])
        commands2.shape:  torch.Size([256, 64, 1])'''
        # print('paramaters.shape',paramaters.shape)
        '''command: [50, 60, 1]  paramaters: [50, 60, 16]'''
        allcommand = torch.cat((commands, paramaters),dim=-1)
        all_command = allcommand[0:10,:,:]
        all_command = all_command.cpu().detach().numpy()
        #save one_command to one_command.txt
        for index, i in enumerate(all_command):
            i.astype(int)
            np.savetxt(f'/scratch/sg7484/CADGen/bulletpoints/mae_cad/decoder_result/{index}.txt',i)
      
        res = { 
            "args_logits": args_logits
        }        
        return res


    def forward(self, command):
        batch_size = command.shape[0]
        x = torch.randn(batch_size, self.embed_dim)
        x = x.to(self.device)
        #repeat x in dimension 1
        x = x.unsqueeze(1).repeat(1, self.max_len, 1)
        pred = self.forward_decoder(command, x)  # [N, L, p*p*3]        
        return pred


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
