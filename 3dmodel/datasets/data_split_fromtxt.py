import os
import numpy as np
txt_path = '/scratch/sg7484/CMDGen/3dmodel/datasets/datalist/hpc/all_data.txt'
save_path ='/scratch/sg7484/CMDGen/3dmodel/datasets/datalist/hpc'
with open(txt_path,'r') as f:
    lines = f.readlines()
length  = len(lines)
idxs = np.random.randint(0, length, size=length)
split_ratio = 0.9
trains = [lines[i] for i in idxs[:int(length*split_ratio)]]
tests = [lines[i] for i in idxs[int(length*split_ratio):]]
print('len(trains): ',len(trains))

print('len(tests): ',len(tests))
for line in trains:
    with open(os.path.join(save_path,'train.txt'),'a') as f:
        f.write(line)
for line in tests:
    with open(os.path.join(save_path,'test.txt'),'a') as f:
        f.write(line)

'''len(trains):  35178
len(tests):  3909'''