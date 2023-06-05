import numpy as np
import torch
from .io import IO
import torch.utils.data as data
import os
import cv2
import torchvision.transforms as transforms
from PIL import Image
from cadlib.macro import *
import h5py
'''data:
        train.txt
        test.txt
        pic:
            00000001(index):
                00000000_f.png
                00000000_t.png
                00000000_r.png'''
class CADGENdataset(data.Dataset):
    def __init__(self,cfg,test):
        self.test = test
        self.data_root= cfg.data_root
        self.cad_root = cfg.cad_root
        self.h5_root = cfg.cmd_root
        self.pic_root = cfg.pic_root
        
        self.train_lis = os.path.join(self.data_root,'train.txt')
        self.test_lis = os.path.join(self.data_root,'test.txt')
        if self.test:
            with open(self.test_lis, 'r') as f:
                lines = f.readlines()
        else:
            with open(self.train_lis, 'r') as f:
                lines = f.readlines()
        self.file_list = []
        for line in lines:
            self.file_list.append(line.strip())

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256, 256]),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.max_total_len  = MAX_TOTAL_LEN
        self.sample_points_num = cfg.npoints
        self.npoints = cfg.n_points
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
    
    def __getitem__(self, index):
        data_num = self.file_list[index]
        cad_path = os.path.join(self.cad_root, data_num+'.npy')
        #print('cad_path: ',cad_path)
        cad_data = IO.get(cad_path).astype(np.float32)
        #print('debug-------------------------------------')
        cad_data = self.random_sample(cad_data, self.sample_points_num)
        #print('debug-------------------------------------')
        cad_data = self.pc_norm(cad_data)
        cad_data = torch.from_numpy(cad_data).float()
        
        h5_path = os.path.join(self.h5_root, data_num+'.h5') 
        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:] # (len, 1 + N_ARGS)
        #print('cad_vec: ',cad_vec)
        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        np.set_printoptions(threshold=np.inf)
        #print('cad_vec: \n',np.array(cad_vec))
        #print('cad_vec.shape: ',cad_vec.shape)
        command = cad_vec[:, 0]
        paramaters = cad_vec[:, 1:]
        command = torch.tensor(command, dtype=torch.long)
        paramaters = torch.tensor(paramaters, dtype=torch.long)
        command = command.clamp(0,5)
        paramaters = paramaters.clamp(-1,255)
        
        front_pic_path = os.path.join(self.pic_root,data_num+'_f.png')
        top_pic_path = os.path.join(self.pic_root,data_num+'_t.png')
        side_pic_path = os.path.join(self.pic_root,data_num+'_r.png')
        front_pic = cv2.imread(front_pic_path)
        top_pic = cv2.imread(top_pic_path)
        side_pic = cv2.imread(side_pic_path)
        front_pic = self.transforms(front_pic)
        top_pic = self.transforms(top_pic)
        side_pic = self.transforms(side_pic)
        # print('front_pic.shape: ',front_pic.shape)
        # print('top_pic.shape: ',top_pic.shape)
        # print('side_pic.shape: ',side_pic.shape)
        # print('cad_data.shape: ',cad_data.shape)
        # print('command.shape: ',command.shape)
        # print('paramaters.shape: ',paramaters.shape)
        # print('data_num: ',data_num)
        data = {'data':(front_pic,top_pic,side_pic,cad_data,command,paramaters),'id':data_num}
        #print('data_num: ',data_num)
        return data
    
    def __len__(self):
        return len(self.file_list)

