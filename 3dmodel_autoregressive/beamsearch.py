import warnings
from torch.nn import functional as F
import logging
from torch import nn
from pprint import pformat
import functools

from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
from utils.misc import *
import time
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datasets.dataset import CADGENdataset
from model.model import Views2Points
from timm.scheduler import CosineLRScheduler
from model.loss import CADLoss
import os
import os, sys
# online package
# optimizer
import torch.optim as optim
import numpy as np
from torch.cuda.amp import autocast as autocast
import h5py
torch.set_printoptions(profile="full")

from cadlib.macro import *
def logits2vec( outputs, refill_pad=True, to_numpy=True):
    """network outputs (logits) to final CAD vector"""
    #print('outputs[command_logits]: ', outputs['command_logits'])
    #print('outputs[args_logits]: ', outputs['args_logits'])
    out_command = torch.argmax(torch.softmax(outputs['command_logits'], dim=-1), dim=-1)  # (N, S)
    out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1  # (N, S, N_ARGS)
    if refill_pad: # fill all unused element to -1
        mask = ~torch.tensor(CMD_ARGS_MASK).bool().cuda()[out_command.long()]
        out_args[mask] = -1

    out_cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)
    if to_numpy:
        out_cad_vec = out_cad_vec.detach().cpu().numpy()
    return out_cad_vec

class BeamHypotheses(object):
    def __init__(self, n_hyp, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def _length_norm(self, length):
        #return length ** self.length_penalty
        # beam search alpha: https://opennmt.net/OpenNMT/translation/beam_search/
        return (5 + length) ** self.length_penalty / (5 + 1) ** self.length_penalty

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        #score = sum_logprobs / len(hyp) ** self.length_penalty
        score = sum_logprobs / self._length_norm(len(hyp))
        #print('score: ',score)
        #print('hyp: ',hyp)
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            # print('done')
            # print('len(self): ',len(self))
            # print('self.worst_score: ',self.worst_score)
            # print('best_sum_logprobs: ',best_sum_logprobs)
            # print('self.max_length: ',self.max_length)
            # print(' best_sum_logprobs / self._length_norm(self.max_length): ', best_sum_logprobs / self._length_norm(self.max_length))
            #return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty
            return self.worst_score >= best_sum_logprobs / self._length_norm(self.max_length)


class GeneratorWithBeamSearch(object):
    def __init__(self,cfg):
        self.num_beams = cfg.num_beam
        self.max_length = MAX_TOTAL_LEN
        self.num_keep_best = 1
        self.device = cfg.device
        self.length_penalty = 0.6
        self.per_node_beam_size = 2
        self.pad_token_id = -1
    def search(self,visual_features,model):
        #print('visual_features.shape: ',visual_features.shape)
        '''torch.Size([1, 64, 256])'''
        device_cpu = 'cpu'
        batch_size = visual_features.size(0)
        input_ids_command = args = torch.ones(self.num_beams*batch_size, 1).fill_(ALL_COMMANDS.index('SOL')). \
            type(torch.long).to(device_cpu)
        input_ids_args = torch.ones(self.num_beams*batch_size,1 , N_ARGS).fill_(PAD_VAL). \
            type(torch.long).to(device_cpu)
        generated_hyps_command = [
            BeamHypotheses(self.num_keep_best, self.max_length, self.length_penalty, early_stopping=False) for _ in range(batch_size)
        ]
        generated_hyps_args = [
            BeamHypotheses(self.num_keep_best, self.max_length, self.length_penalty, early_stopping=False) \
                for _ in range(batch_size) for i in range(N_ARGS)
        ]
        #print('len(generated_hyps_args): ',len(generated_hyps_args))
        beam_scores = torch.zeros((batch_size, self.num_beams), dtype=torch.float, device=device_cpu)
        beam_scores[:, 1:] = -1e9
        beam_scores_command = beam_scores.view(-1).clone()
        beam_scores_args = [beam_scores.view(-1).clone() for _ in range(N_ARGS)]
        #print('beam_scores_command.shape: ',beam_scores_command.shape)
        #print('beam_scores_args.shape: ',beam_scores_args.shape)
        '''beam_scores_command.shape:  torch.Size([4])
            beam_scores_args.shape:  torch.Size([4])'''
        
        done = [False for _ in range(batch_size)]
        #done = [[False for _ in range(self.num_beams)] for _ in range(batch_size)]
        cur_len = 1
        if self.num_beams > 1:
            batch_size, num_token, channels = visual_features.size()
            # shape: (batch_size * beam_size, channels, height, width)
            visual_features = visual_features.unsqueeze(1).repeat(1, self.num_beams, 1, 1)
            visual_features = visual_features.view(
                batch_size * self.num_beams, num_token, channels
            )
        while cur_len < self.max_length:

            #print('visual_features.shape: ',visual_features.shape)
            #print('visual_features.shape: ',visual_features.shape)
            '''visual_features.shape:  torch.Size([4, 64, 256])'''
            #print('input_ids_command.shape: ',input_ids_command.shape)
            #print('input_ids_args.shape: ',input_ids_args.shape)
            '''input_ids_command.shape:  torch.Size([5, 1])
            input_ids_args.shape:  torch.Size([5, 1, 16])'''
            input_ids_command_gpu = input_ids_command.to(self.device)
            input_ids_args_gpu = input_ids_args.to(self.device)
            with torch.no_grad():
                with autocast():
                    score = model(visual_features,input_ids_command_gpu, input_ids_args_gpu)
            #print('score["command_logits"].shape: ',score["command_logits"].shape)
            #print('score["args_logits"].shape: ',score["args_logits"].shape)
            score_command= score["command_logits"][:,-1,:].float().detach().cpu()
            score_args = score["args_logits"][:,-1,:,:].float().detach().cpu()
            output_temp = {'command_logits':score_command,'args_logits':score_args}
            #print('score_command.shape: ',score_command.shape,'score_args.shape: ',score_args.shape)
            '''score_command.shape:  torch.Size([4, 6]) score_args.shape:  torch.Size([4, 16, 257])'''
            command_size = score_command.shape[-1]
            _, args_num, args_size = score_args.shape
            
            score_command = F.log_softmax(score_command, dim=-1)
            #print('score_command: ',score_command)
            temp = beam_scores_command[:, None].expand_as(score_command)
            #print('temp: ',temp)
            #print('temp.shape: ',temp.shape)
            '''temp.shape:  torch.Size([4, 6])'''
            _score_command = score_command + temp
            _score_command = _score_command.view(batch_size, self.num_beams * command_size)
            #print('_score_command: ',_score_command)
            next_scores_command, next_words_command = torch.topk(_score_command, self.per_node_beam_size * self.num_beams, dim=1, largest=True, sorted=True)
            #print('next_scores_command: ',next_scores_command)
            #print('next_words_command: ',next_words_command)
            #print('next_scores_command.shape: ',next_scores_command.shape)
            #print('next_words_command.shape: ',next_words_command.shape)
            '''next_scores_command:  tensor([[-9.8287e-01, -1.1747e+00, -2.3159e+00, -2.5144e+00, -2.6098e+00,
                    -2.7534e+00, -1.0000e+09, -1.0000e+09]], device='cuda:0',
                grad_fn=<TopkBackward>)
            next_words_command:  tensor([[1, 0, 4, 5, 2, 3, 7, 6]], device='cuda:0')
            next_scores_command.shape:  torch.Size([1, 8])
            next_words_command.shape:  torch.Size([1, 8])'''
            next_scores_args = []
            next_words_args = []
            for j in range(N_ARGS):
                score_args[:,j,:] = F.log_softmax(score_args[:,j,:], dim=-1)
                #print('score_args[:,j,:].shape: ',score_args[:,j,:].shape)
                _score_args = score_args[:,j,:] + beam_scores_args[j][:,None].expand_as(score_args[:,j,:])
                #print('_score_args.shape: ',_score_args.shape)
                _score_args = _score_args.view(batch_size, self.num_beams * args_size)
                #print('_score_args2.shape: ',_score_args.shape)
                next_scores_args_one, next_words_args_one = torch.topk(_score_args, self.per_node_beam_size * self.num_beams, dim=1, largest=True, sorted=True)
                #print('next_scores_args_one.shape: ',next_scores_args_one.shape,'next_words_args_one.shape: ',next_words_args_one.shape)
                next_scores_args.append(next_scores_args_one.unsqueeze(1))
                next_words_args.append(next_words_args_one.unsqueeze(1))
            next_scores_args = torch.cat(next_scores_args,dim=1)
            next_words_args = torch.cat(next_words_args,dim=1)
            next_scores_args = next_scores_args.permute(0,2,1)
            next_words_args = next_words_args.permute(0,2,1)            
    
            #print('next_scores_args: ',next_scores_args)
            #print('next_words_args: ',next_words_args)
            '''next_scores_args:  tensor([[[-7.1146e-02, -1.4655e-01, -1.7056e+00, -1.8809e+00, -1.9084e+00,
                    -2.2221e+00, -2.9592e+00, -3.1264e+00],
                    [-1.4161e-01, -1.7486e-01, -2.4963e-01, -1.7216e+00, -1.9211e+00,
                    -2.4229e+00, -2.5414e+00, -2.6657e+00],
                    [-4.3219e-02, -4.2358e-01, -1.7022e+00, -1.7453e+00, -1.8248e+00,
                    -2.3291e+00, -2.6093e+00, -2.8382e+00],
                    [-2.3378e-02, -1.4411e-01, -1.7648e-01, -1.2030e+00, -1.9869e+00,
                    -2.3276e+00, -2.6755e+00, -3.4961e+00],
                    [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09,
                    -1.0000e+09, -1.0000e+09, -1.0000e+09],
                    [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09,
                    -1.0000e+09, -1.0000e+09, -1.0000e+09],
                    [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09,
                    -1.0000e+09, -1.0000e+09, -1.0000e+09],
                    [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09,
                    -1.0000e+09, -1.0000e+09, -1.0000e+09],
                    [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09,
                    -1.0000e+09, -1.0000e+09, -1.0000e+09],
                    [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09,
                    -1.0000e+09, -1.0000e+09, -1.0000e+09],
                    [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09,
                    -1.0000e+09, -1.0000e+09, -1.0000e+09],
                    [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09,
                    -1.0000e+09, -1.0000e+09, -1.0000e+09],
                    [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09,
                    -1.0000e+09, -1.0000e+09, -1.0000e+09],
                    [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09,
                    -1.0000e+09, -1.0000e+09, -1.0000e+09],
                    [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09,
                    -1.0000e+09, -1.0000e+09, -1.0000e+09],
                    [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09,
                    -1.0000e+09, -1.0000e+09, -1.0000e+09]]], device='cuda:0',
                grad_fn=<TopkBackward>)
            next_words_args:  tensor([[[772, 129, 386, 429, 415, 450, 723, 615],
                    [386, 900, 643,  21, 579, 450, 964,  32],
                    [643, 386, 868, 964,  93, 358,  38, 102],
                    [772, 515, 386, 139, 140, 144, 516, 171],
                    [  7,   6,   4,   5,   1,   0,   2,   3],
                    [  7,   6,   4,   5,   1,   0,   2,   3],
                    [  7,   6,   4,   5,   1,   0,   2,   3],
                    [  7,   6,   4,   5,   1,   0,   2,   3],
                    [  7,   6,   4,   5,   1,   0,   2,   3],
                    [  7,   6,   4,   5,   1,   0,   2,   3],
                    [  7,   6,   4,   5,   1,   0,   2,   3],
                    [  7,   6,   4,   5,   1,   0,   2,   3],
                    [  7,   6,   4,   5,   1,   0,   2,   3],
                    [  7,   6,   4,   5,   1,   0,   2,   3],
                    [  7,   6,   4,   5,   1,   0,   2,   3],
                    [  7,   6,   4,   5,   1,   0,   2,   3]]], device='cuda:0')'''
            #print('next_scores_args.shape: ',next_scores_args.shape)
            #print('next_words_args.shape: ',next_words_args.shape)
            '''next_scores_args.shape:  torch.Size([1, 16, 8])
            next_words_args.shape:  torch.Size([1, 16, 8])'''
            next_batch_beam_command = []
            next_batch_beam_args = [[] for _ in range(N_ARGS)]
            for batch_ex in range(batch_size):
                
                #generated_hyps_args, generated_hyps_command 
                #print('done[batch_ex]: ',done[batch_ex])
                #print('done or not: ',generated_hyps_command[batch_ex].is_done(next_scores_command[batch_ex].max().item()))
                # done[batch_ex] = done[batch_ex] or generated_hyps_command[batch_ex].is_done(next_scores_command[batch_ex].max().item())
                # if done[batch_ex]:
                #     print('padding')
                #     next_batch_beam_command.extend([(0, 3, 0)] * self.num_beams)  # pad the batch
                #     for j in range(N_ARGS):
                #         next_batch_beam_args[j].extend([(0, -1, 0)] * self.num_beams)
                #     continue
                next_sent_beam_command = []
                next_sent_beam_args = [[] for _ in range(N_ARGS)]
                #print('next_words_args[batch_ex].shape: ',next_words_args[batch_ex].shape)
                #print('next_scores_args[batch_ex].shape: ',next_scores_args[batch_ex].shape)
                for idx_args, score_args, idx_command, score_command in zip(next_words_args[batch_ex],
                    next_scores_args[batch_ex],next_words_command[batch_ex], next_scores_command[batch_ex]):
                    #print('idx_args.device: ',idx_args.device,'score_args.device: ',score_args.device)
                    #print('idx_command.device: ',idx_command.device,'score_command.device: ',score_command.device)
                    #print('idx_args: ',idx_args, 'score_args: ',score_args,'idx_command: ',idx_command,'score_command: ',score_command)
                    '''idx_args:  tensor([772, 129, 386, 415, 429, 450, 723, 615], device='cuda:0') score_args:  tensor([-0.0727, -0.1393, -1.7029, -1.8953, -1.9092, -2.2081, -2.9198, -3.1349],
                        device='cuda:0', grad_fn=<UnbindBackward>) idx_command:  tensor(1, device='cuda:0') score_command:  tensor(-1.0185, device='cuda:0', grad_fn=<UnbindBackward>)'''
                    #print('idx_command: ',idx_command)
                    # print('idx_args.shape: ',idx_args.shape,'score_args.shape: ',score_args.shape)
                    # print('idx_args: ',idx_args)
                    beam_id_command = idx_command // command_size
                    word_id_command = idx_command % command_size 
                    # print('beam_id_command: ',beam_id_command)
                    # print('word_id_command: ',word_id_command)
                    if word_id_command.item() == 3 or cur_len + 1 == self.max_length:
                        #command_ys, paramaters_ys
                        #print('input_ids_command: ',input_ids_command[batch_ex * self.num_beams + beam_id_command, :cur_len])
                        generated_hyps_command[batch_ex].add(
                            input_ids_command[batch_ex * self.num_beams + beam_id_command, :cur_len].clone(), score_command.item())
                        # print('input_ids_args.shape: ',input_ids_args.shape)
                        # print('score_args.shape: ',score_args.shape)
                        '''input_ids_args.shape:  torch.Size([6, 1, 16])'''
                        for j in range(N_ARGS):
                            beam_id_args = idx_args[j] // args_size
                            # print('paramaters_ys: ',paramaters_ys)
                            # print('paramaters_ys[batch_ex * self.num_beams + beam_id_args , :cur_len, j]: ',paramaters_ys[batch_ex * self.num_beams + beam_id_args , :cur_len, j])
                            # print('paramaters_ys.shape: ',paramaters_ys.shape)
                            generated_hyps_args[batch_ex*j].add(
                            input_ids_args[batch_ex * self.num_beams + beam_id_args , :cur_len, j].clone(), score_args[j].item())
                    else:
                        #print('append')
                        next_sent_beam_command.append((score_command, word_id_command, batch_ex * self.num_beams + beam_id_command))
                        for j in range(N_ARGS):
                            word_id_args = idx_args[j] % args_size
                            beam_id_args = idx_args[j] // args_size
                            next_sent_beam_args[j].append((score_args[j], word_id_args, batch_ex * self.num_beams + beam_id_args))
                    if len(next_sent_beam_command) == self.num_beams:
                        #print('break')
                        break
                if cur_len + 1 == self.max_length:
                    assert len(next_sent_beam_command) == 0
                else:
                    #print('len(next_sent_beam_command): ',len(next_sent_beam_command))
                    assert len(next_sent_beam_command) == self.num_beams
                #print('len(next_sent_beam_command): ',len(next_sent_beam_command))
                if len(next_sent_beam_command) == 0:
                    next_sent_beam_command = [(0, self.pad_token_id, 0)] * self.num_beams  # pad the batch
                    
                next_batch_beam_command.extend(next_sent_beam_command)
                #print('len(next_batch_beam_command): ',len(next_batch_beam_command))
                for j in range(N_ARGS):
                    next_batch_beam_args[j].extend(next_sent_beam_args[j])
                #print('len(next_batch_beam_args[0]): ',len(next_batch_beam_args[0]))
                assert len(next_batch_beam_command) == self.num_beams * (batch_ex + 1)
            #print('next_batch_beam_command: ',next_batch_beam_command)
            '''1:
                generated_hyps_args[0].hyp:  []
                generated_hyps_command[0].hyp:  []
                len(next_batch_beam_command):  5'''
            assert len(next_batch_beam_command) == batch_size * self.num_beams
            #print('beam_scores_command.shape:',beam_scores_command.shape)
            '''beam_scores_command.shape: torch.Size([5])'''
            #add command 
            #print('input_ids_command.shape: ',input_ids_command.shape)
            '''input_ids_command.shape:  torch.Size([5, 1])'''
            #print('next_batch_beam_command: ',next_batch_beam_command)
            beam_scores_command = beam_scores_command.new([x[0] for x in next_batch_beam_command])
            #print('beam_scores_command.shape: ',beam_scores_command.shape)
            '''beam_scores_command.shape:  torch.Size([5])'''
            beam_words_command = input_ids_command.new([x[1] for x in next_batch_beam_command])
            #print('beam_words_command.shape: ',beam_words_command.shape)
            '''beam_words_command.shape:  torch.Size([5])'''
            beam_idx_command = input_ids_command.new([x[2] for x in next_batch_beam_command])
            #print('beam_idx_command.shape: ',beam_idx_command.shape)
            '''beam_idx_command.shape:  torch.Size([5])'''
            #print('input_ids_command.shape: ',input_ids_command.shape)
            '''input_ids_command.shape:  torch.Size([5, 1])'''
            input_ids_command = input_ids_command[beam_idx_command, :]
            #print('input_ids_command.shape.shape: ',input_ids_command.shape)
            '''input_ids_command.shape.shape:  torch.Size([5, 1(change)])'''
            input_ids_command = torch.cat([input_ids_command, beam_words_command.unsqueeze(1)], dim=-1)
            #print('input_ids_command2.shape: ',input_ids_command.shape)
            '''input_ids_command2.shape:  torch.Size([5, 2(change)])'''
            #print('input_ids_command: ',input_ids_command)
            '''input_ids_command:  tensor([[4, 1],
                [4, 0],
                [4, 4],
                [4, 5],
                [4, 2]], device='cuda:0')'''
            beam_words_args = [[] for _ in range(N_ARGS)]
            beam_idx_args = [[] for _ in range(N_ARGS)]
            #add paramaters  
            #print('input_ids_args.shape: ',input_ids_args.shape)
            '''input_ids_args.shape:  torch.Size([5, 1, 16])'''
            input_ids_args = input_ids_args.permute(2,0,1)
            input_ids_args_temp = [[] for _ in range(N_ARGS)]
            #print('input_ids_args.shape: ',input_ids_args.shape)
            #print('len(next_batch_beam_args): ',len(next_batch_beam_args))
            for j in range(N_ARGS):
                beam_scores_args[j] = beam_scores_args[j].new([x[0] for x in next_batch_beam_args[j]])
                #print('beam_scores_args[j]: ',beam_scores_args[j].shape)
                beam_words_args[j] = input_ids_args[j].new([x[1] for x in next_batch_beam_args[j]])
                #print('beam_words_args[j]: ',beam_words_args[j].shape)
                beam_idx_args[j] = input_ids_args[j].new([x[2] for x in next_batch_beam_args[j]])
                #print('beam_idx_args[j].shape: ',beam_idx_args[j].shape)
                
                input_ids_args[j] = input_ids_args[j][beam_idx_args[j], :]
                #print('input_ids_args[j].shape: ',input_ids_args[j].shape)
                temp = beam_words_args[j].unsqueeze(1)
                #print('temp.shape: ',temp.shape)
                '''input_ids.shape:  torch.Size([4, 39(change)])'''
                input_ids_args_temp[j] = torch.cat([input_ids_args[j], beam_words_args[j].unsqueeze(1)], dim=-1)
                input_ids_args_temp[j] = input_ids_args_temp[j].unsqueeze(0)
                #print('input_ids_args2.shape: ',input_ids_args.shape)
                #print('input_ids_args: ',input_ids_args)
            input_ids_args = torch.cat(input_ids_args_temp, dim = 0)
            input_ids_args = input_ids_args.permute(1,2,0)
            # print('input_ids_args.shape: ',input_ids_args.shape)
            # print('input_ids_command.shape: ',input_ids_command.shape)
            print('input_ids_command: ',input_ids_command)
            #print( 'generated_hyps_command[0].hyp: ',generated_hyps_command.hyp)
            #print('input_ids_args: ',input_ids_args)
            cur_len = cur_len + 1
            print('done: ',done)
            if all(done):
                break
        # select the best hypotheses
        tgt_len = torch.ones(batch_size, num_keep_best, dtype=torch.long)
        logprobs = torch.zeros(batch_size, num_keep_best,
                dtype=torch.float).fill_(-1e5).to(input_ids.device)
        all_best = []
            
            
def main():

    cfg = parser.get_args()
    max_len = MAX_TOTAL_LEN
    
    
    model = Views2Points(cfg)
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load from checkpoint if provided
    model_path = '/scratch/sg7484/CMDGen/results/test_autogressive/model/CADGEN_50'
    model_par = torch.load(model_path)
    model.load_state_dict(model_par)
    model = model.to(cfg.device)
    model.eval()
    beamsearch = GeneratorWithBeamSearch(cfg)
    # create dataloader
    dataset_test = CADGENdataset(cfg, test=False)
    test_loader = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=cfg.num_workers)
    #print("Total number of test data:", len(test_loader))

    if not os.path.exists(cfg.test_outputs):
        os.makedirs(cfg.test_outputs)
    #cfg.outputs = "{}/results/test_{}".format(cfg.exp_dir, cfg.ckpt)


    # evaluate
    num_beams = cfg.num_beam
    max_length = MAX_TOTAL_LEN
    num_keep_best = 1
    device = cfg.device
    length_penalty = 0.6
    per_node_beam_size = 2
    pad_token_id = -1
    for i, data in enumerate(test_loader):
        if i ==0:
            continue
        front_pic,top_pic,side_pic,cad_data,command,paramaters, data_num= data
        print('data_id: ', data_num)

        #print('command.shape: ',command.shape)
        #print('paramaters.shape: ',paramaters.shape)
        front_pic = front_pic.to(cfg.device)
        top_pic = top_pic.to(cfg.device)
        side_pic = side_pic.to(cfg.device)
        cad_data = cad_data.to(cfg.device)
        command = command.to(cfg.device)
        paramaters = paramaters.to(cfg.device)
        # print('command[:,:2,:]: ',command[:,:2])
        # print('paramaters[:,:2,:]: ',paramaters[:,:2,:])
        tgt_commands = command[:,1:]
        tgt_paramaters = paramaters[:,1:,:]
        print('tgt_commands: ',tgt_commands)
        # use true data
        # command_ys = command[:,:5]
        # paramaters_ys = paramaters[:,:5,:]
        command_ys = args = torch.ones(1, 1).fill_(ALL_COMMANDS.index('SOL')). \
            type(torch.long).to(cfg.device)
        #print('command_ys.shape: ',command_ys.shape)
        '''command_ys.shape:  torch.Size([1, 1])'''
        paramaters_ys = torch.ones(1, 1, N_ARGS).fill_(PAD_VAL). \
            type(torch.long).to(cfg.device)
        #print('paramaters_ys.shape: ',paramaters_ys.shape)
        '''paramaters_ys.shape:  torch.Size([1, 1, 16])'''
        with torch.no_grad():
            with autocast():
                visual_features = model.forward_encoder(front_pic,top_pic,side_pic,cad_data,command_ys, paramaters_ys)
        #print('z.shape: ',z.shape)
        '''z.shape:  torch.Size([1, 64, 256])'''
                #print('visual_features.shape: ',visual_features.shape)
        '''torch.Size([1, 64, 256])'''
        beamsearch.search(visual_features,model.forward_decoder)
        break
            
if __name__=='__main__':
    main()

            
            