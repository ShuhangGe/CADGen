import shutil
import logging
import os
import numpy as np
train_file = '/scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/train.txt'
test_file = '/scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/test.txt'
with open(train_file, 'r') as f:
    train_files = f.readlines()
with open(test_file, 'r') as f:
    test_files = f.readlines()
# print(len(train_files))
# print(len(test_files))
# print(len(train_files + test_files))
final_files = train_files + test_files
length = len(final_files)
#copy files to the new directory
logging.basicConfig(level=logging.DEBUG,
                    filename='/scratch/sg7484/CADGen/3dmodel_autoregressive/datasets/new.log',
                    filemode='a',  
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
for index, train_file in enumerate(final_files):
    print(f'{index}/{length}')
    if index<=28840:
        continue
    else:
        file_num = train_file.strip()
        file_path_f = f'/scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/data3D_pic_256/abc_{file_num[:4]}_step_v00/{file_num}/{file_num}_f.png'
        file_path_r = f'/scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/data3D_pic_256/abc_{file_num[:4]}_step_v00/{file_num}/{file_num}_r.png'
        file_path_t = f'/scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/data3D_pic_256/abc_{file_num[:4]}_step_v00/{file_num}/{file_num}_t.png'
        if os.path.exists(file_path_f) and os.path.exists(file_path_r) and os.path.exists(file_path_t):

            shutil.copy(file_path_f, '/scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/pic')
            shutil.copy(file_path_r, '/scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/pic')
            shutil.copy(file_path_t, '/scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/pic')
        else:
            logging.info(f'file_num: {file_num}')
