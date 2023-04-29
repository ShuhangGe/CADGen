import os
import numpy as np
data_floder = '/home/shuhang/Desktop/data/cad/Sketch_1_Extrude_1/cmd'
train_ratio = 0.9
names = os.listdir(data_floder)
names.sort()
train_names = names[:int(len(names)*train_ratio)]
test_names = names[int(len(names)*train_ratio):]
print(type(names))
for index, name in enumerate(train_names) :
    name = name[:-3]+'\n'
    with open('/home/shuhang/Desktop/data/cad/Sketch_1_Extrude_1/train.txt','a') as f:
        f.write(name)
for index, name in enumerate(test_names) :
    name = name[:-3]+'\n'
    with open('/home/shuhang/Desktop/data/cad/Sketch_1_Extrude_1/test.txt','a') as f:
        f.write(name)    
