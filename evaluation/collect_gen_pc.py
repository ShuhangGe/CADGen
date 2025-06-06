import os
import glob
import numpy as np
import h5py
from joblib import Parallel, delayed
import argparse
import sys
sys.path.append("..")
from utils import write_ply
from cadlib.visualize import vec2CADsolid, CADsolid2pc


parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default='./save_path', required=True)
parser.add_argument('--n_points', type=int, default=2000)
parser.add_argument('--root_dir', type=str, default='./out')
parser.add_argument('--name', type=str, default='gt_vec')

args = parser.parse_args()
root_dir = args.root_dir
name = args.name
SAVE_DIR = os.path.join(args.src , f'{name}')#out_vec, gt_vec
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def process_one(path):
    data_id = path.split("/")[-1]

    save_path = os.path.join(SAVE_DIR, data_id + ".ply")
    #print(save_path)
    # if os.path.exists(save_path):
    #     return

    # print("[processing] {}".format(data_id))
    with h5py.File(path, 'r') as fp:
        
        out_vec = fp[f"{name}"][:].astype(np.float)#out_vec, gt_vec
        #print(out_vec)
    

    try:
        shape = vec2CADsolid(out_vec)
        #print(shape)
    except Exception as e:
        print("create_CAD failed", data_id)
        return None

    try:
        out_pc = CADsolid2pc(shape, args.n_points, data_id)
        #print('out_pc.shape: ',out_pc.shape)
        '''out_pc.shape:  (2000, 3)'''
    except Exception as e:
        print("convert pc failed:", data_id)
        return None
    
    save_path = os.path.join(SAVE_DIR, data_id + ".ply")
    write_ply(out_pc, save_path)


# all_paths = glob.glob(os.path.join(args.src, "*.h5"))
# Parallel(n_jobs=8, verbose=2)(delayed(process_one)(x) for x in all_paths)
all_path = glob.glob(os.path.join(root_dir, "*.h5"))
# print(all_path)
all_path.sort()
for data_path in all_path:
    print('path: ',data_path)
    # data_path = os.path.join(root_dir,name)
    process_one(data_path)

'''python collect_gen_pc.py --src /home/rl4citygen/gsh/data/result_3pic2cmd/out'''