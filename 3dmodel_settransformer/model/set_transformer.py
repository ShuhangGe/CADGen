import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
import random
from knn_cuda import KNN
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

class RFF(nn.Module):
    """
    Row-wise FeedForward layers.
    """
    def __init__(self, d):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(inplace=True),
            nn.Linear(d, d), nn.ReLU(inplace=True),
            nn.Linear(d, d), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.layers(x)
class Multihead(nn.Module):
    '''
    MultiHeadAttention
    input:
        query --- [N, T_q, query_dim] 
        key --- [N, T_k, key_dim]
        mask --- [N, T_k]
    output:
        out --- [N, T_q, num_units]
        scores -- [h, N, T_q, T_k]
    '''
 
    def __init__(self, query_dim, key_dim, num_units, num_heads):
 
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
 
        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
 
    def forward(self, query, key, mask=None):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)
 
        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
 
        ## score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
 
        ## mask
        if mask is not None:
            ## mask:  [N, T_k] --> [h, N, T_q, T_k]
            mask = mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads,1,querys.shape[2],1)
            scores = scores.masked_fill(mask, -np.inf)
        scores = F.softmax(scores, dim=3)
 
        ## out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        return out

    
class InducedSetAttentionBlock(nn.Module):
    '''ISAB'''
    def __init__(self,encoder_dims,max_total_len,device):
        super().__init__()
        self.encoder_dim = encoder_dims
        self.max_total_len = max_total_len
        self.device= device
        self.attention = Multihead(self.encoder_dim,self.encoder_dim,self.encoder_dim,num_heads=4)
        self.params = nn.Parameter(torch.randn(1, self.max_total_len ,self.encoder_dim).to(self.device))
        #self.params = nn.Parameter(torch.FloatTensor(batchsize, cfg.num_group, cfg.max_total_len).to(cfg.device))
        self.rff = RFF(self.encoder_dim)
        self.layer_norm1 = nn.LayerNorm(self.encoder_dim)
        self.layer_norm2 = nn.LayerNorm(self.encoder_dim)
    def forward(self,input):
        b = input.shape[0]
        I = self.params
        I = I.repeat([b, 1, 1])
        output = self.attention(I,input)
        output = I + output
        H = self.layer_norm1(output)
        H = self.rff(H) + H
        H = self.layer_norm2(H)
        return H

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.ones(50,32,128).to(device)
    model = InducedSetAttentionBlock(128,60,device).to(device)
    out = model(data)
    print(out.shape)
        
            

