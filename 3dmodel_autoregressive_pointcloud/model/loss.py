import torch
import torch.nn as nn
import torch.nn.functional as F
from cadlib.macro import CMD_ARGS_MASK
from cadlib.macro import EOS_IDX, SOL_IDX, EXT_IDX
from model.model_utils import _get_padding_mask, _get_visibility_mask
import numpy as np


# def _get_padding_mask(commands, seq_dim=0, extended=False):
#     with torch.no_grad():
#         #print('commands: ',commands)
#         padding_mask = (commands == EOS_IDX).cumsum(dim=seq_dim) == 0
#         #print('padding_mask: ',padding_mask)

#         if extended:
#             # padding_mask doesn't include the final EOS, extend by 1 position to include it in the loss
#             S = commands.size(seq_dim)
#             torch.narrow(padding_mask, seq_dim, 3, S-3).add_(torch.narrow(padding_mask, seq_dim, 0, S-3)).clamp_(max=1)

#         if seq_dim == 0:
#             return padding_mask.unsqueeze(-1)
#         return padding_mask

# def _get_visibility_mask(commands, seq_dim=0):
#     """
#     Args:
#         commands: Shape [S, ...]
#     """
#     S = commands.size(seq_dim)
#     with torch.no_grad():
#         visibility_mask = (commands == EOS_IDX).sum(dim=seq_dim) < S - 1

#         if seq_dim == 0:
#             return visibility_mask.unsqueeze(-1)
#         return visibility_mask
    
class CADLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_commands = cfg.n_commands
        self.args_dim = cfg.args_dim + 1
        self.weights = cfg.loss_weights

        self.register_buffer("cmd_args_mask", torch.tensor(CMD_ARGS_MASK))

    def forward(self, output):
        # Target & predictions
        tgt_commands, tgt_args = output["tgt_commands"], output["tgt_args"]

        visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
        padding_mask = _get_padding_mask(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)

        command_logits, args_logits = output["command_logits"], output["args_logits"]

        mask = self.cmd_args_mask[tgt_commands.long()]
        if (torch.isnan(command_logits).any() or torch.isnan(command_logits).any()) or (torch.isnan(args_logits).any() or torch.isnan(args_logits).any()):
            print('\n','nanananan','\n')

        loss_cmd = F.cross_entropy(command_logits[padding_mask.bool()].reshape(-1, self.n_commands), tgt_commands[padding_mask.bool()].reshape(-1).long().clamp(0, 6),ignore_index=-1)
        #loss_cmd = F.cross_entropy(command_logits.reshape(-1, self.n_commands), tgt_commands.reshape(-1).long().clamp(0, 6),ignore_index=-1)

        np.set_printoptions(threshold=np.inf)
        loss_args = F.cross_entropy(args_logits[mask.bool()].reshape(-1, self.args_dim), (tgt_args[mask.bool()].reshape(-1).long() + 1).clamp(0, 256),ignore_index=-1)  # shift due to -1 PAD_VAL

        loss_cmd = self.weights["loss_cmd_weight"] * loss_cmd
        loss_args = self.weights["loss_args_weight"] * loss_args
        print('loss_cmd: ',loss_cmd, 'loss_args: ',loss_args)
        '''
        mask_token.shape:  torch.Size([50, 19, 128])
        x_full.shape:  torch.Size([50, 32, 128])
        pos_full.shape:  torch.Size([50, 32, 128])
        out_logits[0].shape:  torch.Size([50, 60, 6])
        out_logits[1].shape:  torch.Size([50, 60, 16, 257])'''
        res = {"loss_cmd": loss_cmd, "loss_args": loss_args}
        return res
