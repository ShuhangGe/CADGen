import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from utils.logger import *
import h5py
from .macro import *
class Point2CAD_dataset(data.Dataset):
    def __init__(self, config, test):
        self.data_path = config.data_path
        self.npoints = config.n_points
        self.test = test
        self.data_root= config.data_root
        self.data_list_file = os.path.join(self.data_path, 'train.txt')
        test_data_list_file = os.path.join(self.data_path, 'test.txt')
        self.max_total_len  = MAX_TOTAL_LEN
        self.sample_points_num = config.npoints


        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
            if self.test:
                with open(test_data_list_file, 'r') as f:
                    lines = f.readlines()
            else:
                with open(self.data_list_file, 'r') as f:
                    lines = f.readlines()
        self.file_list = []
        for line in lines:
            self.file_list.append({
                'file_path': line.strip()
            })
        #print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'ShapeNet-55')

        self.permutation = np.arange(self.npoints)
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        #print('sample.shape: ',sample['file_path'])
        #print(sample)
        cad_path = os.path.join(os.path.join(self.data_root,'cad'), sample['file_path']+'.npy')
        #print('cad_path: ',cad_path)
        cad_data = IO.get(cad_path).astype(np.float32)
        cad_data = self.random_sample(cad_data, self.sample_points_num)
        cad_data = self.pc_norm(cad_data)
        cad_data = torch.from_numpy(cad_data).float()

        h5_path = os.path.join(os.path.join(self.data_root,'cmd'), sample['file_path']+'.h5')
        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:] # (len, 1 + N_ARGS)
        #print('cad_vec: ',cad_vec)
        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        np.set_printoptions(threshold=np.inf)
        #print('cad_vec: \n',np.array(cad_vec))
        #print('cad_vec.shape: ',cad_vec.shape)
        command = cad_vec[:, 0]
        args = cad_vec[:, 1:]
        command = torch.tensor(command, dtype=torch.long)
        args = torch.tensor(args, dtype=torch.long)
        #return {"command": command, "args": args}

        data = {'data':(command, args, cad_data),'id':sample['file_path']}
        #print('data: ',data)
        return data

    def __len__(self):
        return len(self.file_list)
    

# import os
# import torch
# import numpy as np
# import torch.utils.data as data
# from .io import IO
# from .build import DATASETS
# from utils.logger import *

# @DATASETS.register_module()
# class ShapeNet(data.Dataset):
#     def __init__(self, config):
#         self.data_root = config.DATA_PATH
#         self.pc_path = config.PC_PATH
#         self.subset = config.subset
#         self.npoints = config.N_POINTS
        
#         self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
#         test_data_list_file = os.path.join(self.data_root, 'test.txt')
        
#         self.sample_points_num = config.npoints
#         self.whole = config.get('whole')

#         print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'ShapeNet-55')
#         print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'ShapeNet-55')
#         with open(self.data_list_file, 'r') as f:
#             lines = f.readlines()
#         if self.whole:
#             with open(test_data_list_file, 'r') as f:
#                 test_lines = f.readlines()
#             print_log(f'[DATASET] Open file {test_data_list_file}', logger = 'ShapeNet-55')
#             lines = test_lines + lines
#         self.file_list = []
#         for line in lines:
#             line = line.strip()
#             taxonomy_id = line.split('-')[0]
#             model_id = line.split('-')[1].split('.')[0]
#             self.file_list.append({
#                 'taxonomy_id': taxonomy_id,
#                 'model_id': model_id,
#                 'file_path': line
#             })
#         print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'ShapeNet-55')

#         self.permutation = np.arange(self.npoints)
#     def pc_norm(self, pc):
#         """ pc: NxC, return NxC """
#         centroid = np.mean(pc, axis=0)
#         pc = pc - centroid
#         m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
#         pc = pc / m
#         return pc
        

#     def random_sample(self, pc, num):
#         np.random.shuffle(self.permutation)
#         pc = pc[self.permutation[:num]]
#         return pc
        
#     def __getitem__(self, idx):
#         sample = self.file_list[idx]

#         data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)

#         data = self.random_sample(data, self.sample_points_num)
#         data = self.pc_norm(data)
#         data = torch.from_numpy(data).float()
#         return sample['taxonomy_id'], sample['model_id'], data

#     def __len__(self):
#         return len(self.file_list)