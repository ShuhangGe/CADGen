import os
import numpy as np
txt_path = '/home/rl4citygen/gsh/data/CMDGen/all_data.txt'
save_path ='/home/rl4citygen/gsh/data/CMDGen'
with open(txt_path,'r') as f:
    lines = f.readlines()
length  = len(lines)
split_ratio = 0.9
trains = lines[:int(length*split_ratio)]
tests = lines[int(length*split_ratio):]
for line in trains:
    with open(os.path.join(save_path,'train.txt'),'a') as f:
        f.write(line)
for line in tests:
    with open(os.path.join(save_path,'test.txt'),'a') as f:
        f.write(line)

    