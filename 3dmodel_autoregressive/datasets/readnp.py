import numpy as np
import glob
import os
import logging  
logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                    filename='new.log',
                    filemode='a',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志, a是追加模式，默认如果不写的话，就是追加模式
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'# 日志格式
                    )
root_path = '/scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/cad'
cad_paths = glob.glob(os.path.join(root_path, '*.npy'))
length = len(cad_paths)
for index, cad_path in enumerate(cad_paths):
    print(f'{index}/{length}')
    try:
        cad_data = np.load(cad_path, allow_pickle=True).astype(np.float32)
    except Exception as e:
        logging.info(f'cad_path: {cad_path}')
        continue
    