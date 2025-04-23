import numpy as np
import os

data_dir = '/home/shuhang/Desktop/data/cad/Sketch_1_Extrude_1_txt'
save_dir = '/home/shuhang/Desktop/data/cad/Sketch_1_Extrude_1/cad'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
names = os.listdir(data_dir)
names.sort()
num = 0
for index,name in enumerate(names) :
    data_path = os.path.join(data_dir,name)
    with open(data_path, "r") as f:  # 打开文件
        #data = f.read()  # 读取文件
        data = f.readlines()
        all = []
        for line in data:
            line = line.split(' ')
            line = np.array(line)
            line = line.astype(np.float)
            all.append(line)
        all = np.array(all)
        #print(type(data))
        name_new = name[:-4]
        save_path = os.path.join(save_dir,name_new)
        print(save_path)
        np.save(save_path,all)
        print(index)
