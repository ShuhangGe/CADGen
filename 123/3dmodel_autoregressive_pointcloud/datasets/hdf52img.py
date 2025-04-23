import numpy as np
import gym
import matplotlib.pyplot as plt
import cv2
import matplotlib.lines as lines
import random
import os
from gym.utils import seeding
import sys
import h5py
cad_dir = '/scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/cad/00005885.npy'
cmd_dir = '/scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/cmd/00005885.h5'
cad_path = os.path.join(cad_dir)
#print('cad_path: ',cad_path)
cad_data = IO.get(cad_path).astype(np.float32)
#print('debug-------------------------------------')
cad_data = random_sample(cad_data, sample_points_num)
#print('debug-------------------------------------')
cad_data = pc_norm(cad_data)
cad_data = torch.from_numpy(cad_data).float()
cad_data = cad_data.to(device)
cad_data = cad_data.unsqueeze(0)


h5_path = os.path.join(cmd_dir) 
with h5py.File(h5_path, "r") as fp:
    cad_vec = fp["vec"][:] # (len, 1 + N_ARGS)
#print('cad_vec: ',cad_vec)
pad_len = max_total_len - cad_vec.shape[0]
cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
np.set_printoptions(threshold=np.inf)
#print('cad_vec: \n',np.array(cad_vec))
#print('cad_vec.shape: ',cad_vec.shape)
command = cad_vec[:, 0]
paramaters = cad_vec[:, 1:]
command = torch.tensor(command, dtype=torch.long)
paramaters = torch.tensor(paramaters, dtype=torch.long)
command = command.clamp(0, 256).to(device)
paramaters = paramaters.clamp(-1 256).to(device)
command = command.unsqueeze(0)
paramaters = paramaters.unsqueeze(0)
print('command: ',command)
print('paramaters: ',paramaters)

'''
side((x),y,z)
front(x,(y),z)
top:(x,y,(z))
'''
model = Views2Points(cfg).to(device)
print(model)
a= model(side,front,top,cad_data,command, paramaters)
        