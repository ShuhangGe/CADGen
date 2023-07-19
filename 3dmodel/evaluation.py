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
parser.add_argument('--src', type=str, default=None, required=True)
parser.add_argument('--n_points', type=int, default=2000)
args = parser.parse_args()
args.src = '/home/rl4citygen/gsh/data/CMDGen/out/fulldata_noauto_deformable3_res18_unet4_best_35_4.13_test_4.85_1e-4_test/outvec_withgt'
SAVE_DIR = '/home/rl4citygen/gsh/data/CMDGen/out/fulldata_noauto_deformable3_res18_unet4_best_35_4.13_test_4.85_1e-4_test'
SAVE_DIR = os.path.join(SAVE_DIR,'gt_vec')#out_vec,gt_vec
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def process_one(path):
    data_id = path.split("/")[-1]

    save_path = os.path.join(SAVE_DIR, data_id + ".ply")
    if os.path.exists(save_path):
        return

    # print("[processing] {}".format(data_id))
    with h5py.File(path, 'r') as fp:
        out_vec = fp["gt_vec"][:].astype(np.float)#out_vec,gt_vec

        np.savetxt(os.path.join(SAVE_DIR, data_id + ".txt"),out_vec,fmt='%1.2f')
    try:
        shape = vec2CADsolid(out_vec)

    except Exception as e:
        print("create_CAD failed", data_id)
        return None

    try:
        out_pc = CADsolid2pc(shape, args.n_points, data_id)
    except Exception as e:
        print("convert pc failed:", data_id)
        return None
    #print(save_path)
    save_path = os.path.join(SAVE_DIR, data_id + ".ply")
    write_ply(out_pc, save_path)


all_paths = glob.glob(os.path.join(args.src, "*.h5"))
Parallel(n_jobs=8, verbose=2)(delayed(process_one)(x) for x in all_paths)
'''[Parallel(n_jobs=8)]: Using backend LokyBackend with 8 concurrent workers.
create_CAD failed 2900_vec.h5
create_CAD failed 2633_vec.h5
create_CAD failed 93_vec.h5
create_CAD failed 714_vec.h5
create_CAD failed 2957_vec.h5
create_CAD failed 615_vec.h5
create_CAD failed 2707_vec.h5
create_CAD failed 413_vec.h5
create_CAD failed 3681_vec.h5
create_CAD failed 712_vec.h5
create_CAD failed 3649_vec.h5
create_CAD failed 1865_vec.h5
create_CAD failed 61_vec.h5
create_CAD failed 4694_vec.h5
create_CAD failed 1877_vec.h5
create_CAD failed 4348_vec.h5
create_CAD failed 133_vec.h5
create_CAD failed 559_vec.h5
create_CAD failed 3187_vec.h5
create_CAD failed 70_vec.h5
create_CAD failed 2718_vec.h5
create_CAD failed 4078_vec.h5
create_CAD failed 4256_vec.h5
create_CAD failed 1867_vec.h5
create_CAD failed 3519_vec.h5
create_CAD failed 1506_vec.h5
create_CAD failed 1533_vec.h5
create_CAD failed 1863_vec.h5
create_CAD failed 2744_vec.h5
create_CAD failed 4409_vec.h5
create_CAD failed 3143_vec.h5
create_CAD failed 123_vec.h5
create_CAD failed 1340_vec.h5
create_CAD failed 1537_vec.h5
create_CAD failed 2347_vec.h5
create_CAD failed 927_vec.h5
create_CAD failed 3858_vec.h5
Warning: 2 faces have been skipped due to null triangulation
[Parallel(n_jobs=8)]: Done  85 out of 100 | elapsed:    3.0s remaining:    0.5s
create_CAD failed 5062_vec.h5
[Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:    3.1s finished'''

#gt:
'''[Parallel(n_jobs=8)]: Done  25 tasks      | elapsed:    2.8s
create_CAD failed 633_vec.h5
[Parallel(n_jobs=8)]: Done  85 out of 100 | elapsed:    3.6s remaining:    0.6s
create_CAD failed 2296_vec.h5'''