from typing import List, Optional

import torch
import torch.nn.modules as nn
import torch.nn.functional as F


def check_para(file_name,**kwargs):
    if file_name is not None:
        print(f'output from {file_name}')
    output =''
    for key in kwargs:
        if kwargs[key] is None:
            temp = f'{key} is None\n'
        elif isinstance(kwargs[key],list):
            temp=''
            if torch.is_tensor(kwargs[key][0]):
                for idex,i in enumerate(kwargs[key]) :
                    temp_list = f'{key}[{idex}]: {kwargs[key][idex].shape}  '
                    temp +=temp_list
                temp+='\n'
        elif isinstance(kwargs[key],int):
            temp = f'{key}: {kwargs[key]}\n'
        elif isinstance(kwargs[key],dict):
            temp=''
            for key_sub in kwargs[key]:
                if kwargs[key][key_sub] is None:
                    temp_dict = f'{key}[{key_sub}] is None    '
                else:
                    temp_dict = f'{key}[{key_sub}]: {kwargs[key][key_sub].shape}    '
                temp +=temp_dict
            temp+='\n'
        else:
            temp = f'{key}: {kwargs[key].shape}\n'
        output =output+temp
    output = output.strip()
    print(output)
def generate_ref_points(width: int,
                        height: int):
    grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    grid_y = grid_y / (height - 1)
    grid_x = grid_x / (width - 1)

    grid = torch.stack((grid_x, grid_y), 2).float()
    grid.requires_grad = False
    return grid


def restore_scale(width: int,
                  height: int,
                  ref_point: torch.Tensor):
    new_point = ref_point.clone().detach()
    new_point[..., 0] = new_point[..., 0] * (width - 1)
    new_point[..., 1] = new_point[..., 1] * (height - 1)

    return new_point


class DeformableHeadAttention(nn.Module):
    def __init__(self, h,
                 d_model,
                 k,
                 last_point_num,
                 scales=1,
                 dropout=0.1,
                 need_attn=False):
        """
        :param h: number of self attention head
        :param d_model: dimension of model
        :param dropout:
        :param k: number of keys
        """
        super(DeformableHeadAttention, self).__init__()
        assert h == 8  # currently header is fixed 8 in paper
        assert d_model % h == 0
        # We assume d_v always equals d_k, d_q = d_k = d_v = d_m / h
        self.d_k = int(d_model / h)
        self.h = h

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

        # self.scales_hw = []
        # for i in range(scales):
        #     self.scales_hw.append([last_feat_height * 2 ** i,
        #                            last_feat_width * 2 ** i])

        self.dropout = None
        if self.dropout:
            self.dropout = nn.Dropout(p=dropout)

        self.k = k
        self.scales = scales
        self.last_point_num = last_point_num
        self.voxel_length = round(last_point_num**(1/3))
  
        self.offset_dims = 3 * self.h * self.k * self.scales
        self.A_dims = self.h * self.k * self.scales

        # 2MLK for offsets MLK for A_mlqk
        self.offset_proj = nn.Linear(d_model, self.offset_dims)
        self.A_proj = nn.Linear(d_model, self.A_dims)

        self.wm_proj = nn.Linear(d_model, d_model)
        self.need_attn = need_attn
    #     self.reset_parameters()

    # def reset_parameters(self):
    #     torch.nn.init.constant_(self.offset_proj.weight, 0.0)
    #     torch.nn.init.constant_(self.A_proj.weight, 0.0)

    #     torch.nn.init.constant_(self.A_proj.bias, 1 / (self.scales * self.k))

    #     def init_xy(bias, x, y):
    #         torch.nn.init.constant_(bias[:, 0], float(x))
    #         torch.nn.init.constant_(bias[:, 1], float(y))

    #     # caution: offset layout will be  M, L, K, 2
    #     bias = self.offset_proj.bias.view(self.h, self.scales, self.k, 3)

    #     init_xy(bias[0], x=-self.k, y=-self.k)
    #     init_xy(bias[1], x=-self.k, y=0)
    #     init_xy(bias[2], x=-self.k, y=self.k)
    #     init_xy(bias[3], x=0, y=-self.k)
    #     init_xy(bias[4], x=0, y=self.k)
    #     init_xy(bias[5], x=self.k, y=-self.k)
    #     init_xy(bias[6], x=self.k, y=0)
    #     init_xy(bias[7], x=self.k, y=self.k)

    def forward(self,
                query: torch.Tensor,
                keys: List[torch.Tensor],
                ref_point: torch.Tensor,
                query_mask: torch.Tensor = None,
                key_masks: Optional[torch.Tensor] = None,
                ):
        """
        :param key_masks:
        :param query_mask:
        :param query: B, H, W, C
        :param keys: List[B, H, W, C]
        :param ref_point: B, H, W, 2
        :return:
        """
        #check_para('deformable_attn',query = query, keys = keys,ref_point = ref_point,query_mask = query_mask, key_masks=key_masks)
        #print('\n')
        #print('self.voxel_length: ',self.voxel_length)
        if key_masks is None:
            key_masks = [None] * len(keys)
        
        assert len(keys) == self.scales

        attns = {'attns': None, 'offsets': None}
        #print('query.shape : ',query.shape )
        #print('len(query.size()): ',len(query.size()))
        if len(query.size())==4:
            nbatches, point_num, _,dim = query.shape    
            query = query.view(nbatches, point_num, dim) 
            nbatches, point_num, _,dim = ref_point.shape
            ref_point = ref_point.view(nbatches, point_num, dim) 
        #print('query.shape: {}, {}'.format(nbatches,point_num))   
        # B, H, W, C
        query = self.q_proj(query)
        #check_para(None,query = query)
        # B, H, W, 2MLK
        offset = self.offset_proj(query)
        #check_para(None,offset = offset)
        # B, H, W, M, 2LK
        offset = offset.view(nbatches, point_num, self.h, -1)
        # B, H, W, MLK
        A = self.A_proj(query)
        #check_para(None,A = A,offset2 = offset)
        # B, H, W, 1, mask before softmax
        #print('\n')
        #print('query_mask is None')
        if query_mask is not None:
            #print('query_mask is not None')
            query_mask_ = query_mask.unsqueeze(dim=-1)
            #######check_para(None,query_mask_ = query_mask_)
            _, _, _, mlk = A.shape
            query_mask_ = query_mask_.expand(nbatches, query_height, query_width, mlk)
            #check_para(None,query_mask_2 = query_mask_)
            A = torch.masked_fill(A, mask=query_mask_, value=float('-inf'))
            #check_para(None,A = A)

        # B, H, W, M, LK
        A = A.view(nbatches, point_num, self.h, -1)
        #check_para(None,A2 = A)
        A = F.softmax(A, dim=-1)
        #check_para(None,A3 = A)
        # mask nan position
        if query_mask is not None:
            # B, H, W, 1, 1
            query_mask_ = query_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            A = torch.masked_fill(A, query_mask_.expand_as(A), 0.0)

        if self.need_attn:
            attns['attns'] = A
            attns['offsets'] = offset

        offset = offset.view(nbatches, point_num, self.h, self.scales, self.k, 3)
        #check_para(None,offset = offset)
        offset = offset.permute(0, 2,3,4,1,5).contiguous()
        #check_para(None,offset = offset)
        # B*M, L, K, H, W, 2
        offset = offset.view(nbatches * self.h, self.scales, self.k, point_num, 3)
        #check_para(None,offset = offset)

        A = A.permute(0, 2, 1, 3).contiguous()
        #check_para(None,A = A)
        # B*M, H*W, LK
        A = A.view(nbatches * self.h, point_num, -1)
        #check_para(None,A = A)

        scale_features = []
        for l in range(self.scales):
            #print('l:   ',l)
            feat_map = keys[l]
            #check_para(None,feat_map = feat_map)
            _, h, w, z,_ = feat_map.shape
            #print('feat_map.shape: ',feat_map.shape)
            key_mask = key_masks[l]
            #check_para(None,key_mask = key_mask)
            # B, H, W, 2
            reversed_ref_point = restore_scale(height=h, width=w, ref_point=ref_point)
            #check_para(None,reversed_ref_point = reversed_ref_point)

            # B, H, W, 2 -> B*M, H, W, 2
            reversed_ref_point = reversed_ref_point.repeat(self.h, 1, 1)
            #check_para(None,reversed_ref_point = reversed_ref_point)
            # B, h, w, M, C_v
            scale_feature = self.k_proj(feat_map).view(nbatches, h, w, z, self.h, self.d_k)
            #check_para(None,scale_feature = scale_feature)
            # if key_mask is not None:
            #     print('key_mask is not None')
            #     # B, h, w, 1, 1
            #     key_mask = key_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            #     #check_para(None,key_mask = key_mask)
            #     key_mask = key_mask.expand(nbatches, h, w, z, self.h, self.d_k)
            #     #check_para(None,key_mask = key_mask)
            #     scale_feature = torch.masked_fill(scale_feature, mask=key_mask, value=0)
            #     #check_para(None,scale_feature = scale_feature)

            # B, M, C_v, h, w
            scale_feature = scale_feature.permute(0, 4, 5, 1, 2,3).contiguous()
            #check_para(None,scale_feature = scale_feature)
            # B*M, C_v, h, w
            scale_feature = scale_feature.view(-1, self.d_k, h, w, z)
            #check_para(None,scale_feature = scale_feature)
            k_features = []
            #print('self.k: ',self.k)
            for k in range(self.k):
                # print('k:   ',k)
                # print('reversed_ref_point.shape: ',reversed_ref_point.shape)
                # print('offset[:, l, k, :, :].shape: ',offset[:, l, k, :, :].shape)
                points = reversed_ref_point + offset[:, l, k, :, :]
                #check_para(None,points = points)
                vgrid_x = 3.0 * points[:, :, 0] / max(w - 1, 1) - 1.0
                #check_para(None,vgrid_x = vgrid_x)
                vgrid_y = 3.0 * points[:, :, 1] / max(h - 1, 1) - 1.0
                #check_para(None,vgrid_y = vgrid_y)
                vgrid_z = 3.0 * points[:, :, 1] / max(z - 1, 1) - 1.0
                #check_para(None,vgrid_z = vgrid_z)
                vgrid_scaled = torch.stack((vgrid_x, vgrid_y, vgrid_z), dim=2)
                num,_,_ = vgrid_scaled.shape
                vgrid_scaled = vgrid_scaled.view(num, self.voxel_length,self.voxel_length,self.voxel_length,3)
                #check_para(None,vgrid_scaled = vgrid_scaled)

                # B*M, C_v, H, W
                feat = F.grid_sample(scale_feature, vgrid_scaled, mode='bilinear', padding_mode='zeros')
                #check_para(None,feat = feat)
                k_features.append(feat)
                #check_para(None,k_features = k_features)
            # B*M, k, C_v, H, W
            k_features = torch.stack(k_features, dim=1)
            #check_para(None,k_features = k_features)
            scale_features.append(k_features)
            #check_para(None,scale_features = scale_features)

        # B*M, L, K, C_v, H, W
        scale_features = torch.stack(scale_features, dim=1)
        #check_para(None,scale_features = scale_features)
        # B*M, H*W, C_v, LK
        scale_features = scale_features.permute(0, 4, 5, 6, 3, 1, 2).contiguous()
        #check_para(None,scale_features = scale_features)
        scale_features = scale_features.view(nbatches * self.h, point_num, self.d_k, -1)
        #check_para(None,scale_features = scale_features)

        # B*M, H*W, C_v
        #print('A.shape: ',A.shape)
        feat = torch.einsum('nlds, nls -> nld', scale_features, A)
        #check_para(None,feat = feat)
        # B*M, H*W, C_v -> B, M, H, W, C_v
        feat = feat.view(nbatches, self.h, point_num, self.d_k)
        #check_para(None,feat = feat)
        # B, M, H, W, C_v -> B, H, W, M, C_v
        feat = feat.permute(0, 2, 1, 3).contiguous()
        #check_para(None,feat = feat)
        # B, H, W, M, C_v -> B, H, W, M * C_v
        feat = feat.view(nbatches, point_num, self.d_k * self.h)
        #check_para(None,feat = feat)

        feat = self.wm_proj(feat)
        #check_para(None,feat = feat)
        if self.dropout:
            feat = self.dropout(feat)
        feat = feat.unsqueeze(2)
        #check_para(None,feat = feat)
        #check_para(None,attns = attns)
        return feat, attns
    

if __name__=='__main__':
    attention = DeformableHeadAttention( h=8,
                 d_model=256,
                 k=3,
                last_point_num=64,
                 scales=3)
    query = torch.randn(3, 64, 256)#3, 100, 1, 256
    keys = [torch.randn(3, 4, 4, 4, 256),torch.randn(3, 8, 8, 8, 256),torch.randn(3, 16, 16, 16, 256)]
    ref_point = torch.randn(3, 64, 3)
    key_masks = [torch.zeros(3, 4, 4, 4),torch.zeros(3, 8, 8, 8),torch.zeros(3, 16, 16, 16)]
    out = attention(query = query,
                keys = keys,
                ref_point = ref_point,
                query_mask = None,
                key_masks = None,)
