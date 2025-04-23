# import os
# import numpy as np
# data_root = '/scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/cmd'
# save_path = '/scratch/sg7484/CMDGen/3dmodel/datasets'
# names = os .listdir(data_root)
# names.sort()
# nums  = []
# for name in names:
#     nums.append(name[:-3])
# #print(nums)
# root = '/scratch/sg7484/data/CMDGen/data3D_pic'
# file_pathes = []
# for num in nums:
#     file_path = root+'/'+f'abc_{num[:4]}_step_v00'+'/'+num
#     file_pathes.append(file_path)
#     with open(os.path.join(save_path, 'all_data.txt'),'a') as f:
#         f.write(file_path+'\n')

import glob
import os

data_root = '/scratch/sg7484/data/CMDGen/all_data/cad_vec'
save_path = '/scratch/sg7484/data/CMDGen/all_data'

all_paths = glob.iglob(data_root+'/*/*.h5')
for path in all_paths:
    with open(os.path.join(save_path, 'all_data.txt'),'a') as f:
        f.write(path+'\n')