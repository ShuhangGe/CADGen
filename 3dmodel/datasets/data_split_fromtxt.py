import os
import numpy as np
import random

txt_path = '/scratch/sg7484/CMDGen/3dmodel/datasets/datalist/hpc/all_data.txt'
save_path ='/scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1'
with open(txt_path,'r') as f:
    lines = f.readlines()
all_lines = []
for line in lines:
    line  = line.split('/')[-1]
    all_lines.append(line)
all_lines.sort()
print(all_lines[:100],'\n')
length  = len(all_lines)
index_order = np.arange(length)
split_ratio = 0.9
index = np.random.permutation(index_order)

train_index = index[:int(length*split_ratio)]
test_index = index[int(length*split_ratio):]
print(train_index[:100],'\n')
print(test_index[:100],'\n')
all_lines = np.array(all_lines)
all_lines = all_lines[index]
train = all_lines[train_index]
test = all_lines[test_index]
train.sort()
test.sort()
num = 0
for item in lines:
    if item == 61:
        num+=1
print("num: ",num)
for line_trains in train:
    with open(os.path.join(save_path,'train.txt'),'a') as f_trains:
            #print('line_trains: ',line_trains)
        f_trains.write(line_trains)
with open(os.path.join(save_path,'test.txt'),'w') as f_tests:
    for line_test in test:
            f_tests.write(line_test)


    