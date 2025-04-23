import os
import numpy as np
import random
txt_path = '/scratch/sg7484/data/CMDGen/all_data/all_data.txt'
save_path ='/scratch/sg7484/data/CMDGen/all_data'
with open(txt_path,'r') as f:
    lines = f.readlines()
length  = len(lines)
random.shuffle(lines)
print(length)
print(type(lines))
split_ratio = 0.9
trains = lines[:int(length*split_ratio)]
tests = lines[int(length*split_ratio):]
for line in trains:
    with open(os.path.join(save_path,'train.txt'),'a') as f:
        f.write(line)
for line in tests:
    with open(os.path.join(save_path,'test.txt'),'a') as f:
        f.write(line)

    