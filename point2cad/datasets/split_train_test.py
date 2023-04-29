import os
import numpy as np
from shutil import copyfile
data_dir = '/home/shuhang/Desktop/point_mae/Point-MAE-main/data/Sketch_1_Extrude_1_np_43908'
save_dir = '/home/shuhang/Desktop/point_mae/Point-MAE-main/data'
train_save_dir = save_dir +'/train'
test_save_dir = save_dir +'/test'
split_ratio = 0.1
if not os.path.exists(train_save_dir):
    os.makedirs(train_save_dir)
if not os.path.exists(test_save_dir):
    os.makedirs(test_save_dir)
names = os.listdir(data_dir)
names.sort()
train_length = int(0.9 *len(names))
names_train = names[:train_length]
names_test = names[train_length:]
for name in names_train:
    data_path = os.path.join(data_dir,name)
    save_path = os.path.join(train_save_dir,name)
    copyfile(data_path,save_path)
for name in names_test:
    data_path = os.path.join(data_dir,name)
    save_path = os.path.join(test_save_dir,name)
    copyfile(data_path,save_path)
