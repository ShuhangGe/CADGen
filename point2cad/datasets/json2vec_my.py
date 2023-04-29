import os
import json
import numpy as np
import h5py
from joblib import Parallel, delayed
import sys
sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.macro import *
#convert pointcloud to commend
name_ROOT = "/home/shuhang/Desktop/data/cad/Sketch_1_Extrude_1/cad"
data_root = '/home/shuhang/Desktop/data/cad/splited_json/Sketch_1_Extrude_1'
#RAW_DATA = os.path.join(DATA_ROOT, "cad_json")
#RECORD_FILE = '/home/rl4citygen/gsh/DeepCAD/DeepCAD-master/data/splited_json/Sketch_1_Extrude_1'

SAVE_DIR = '/home/shuhang/Desktop/data/cad/Sketch_1_Extrude_1/cmd'
#print(SAVE_DIR)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


def process_one(data_id,data_path):
    json_path = data_path
    with open(json_path, "r") as fp:
        data = json.load(fp)

    try:
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        cad_seq.numericalize()
        cad_vec = cad_seq.to_vector(MAX_N_EXT, MAX_N_LOOPS, MAX_N_CURVES, MAX_TOTAL_LEN, pad=False)

    except Exception as e:
        print("failed:", data_id)
        return

    if  (cad_vec is None) or MAX_TOTAL_LEN < cad_vec.shape[0]:
        print("exceed length condition:", data_id)#, cad_vec.shape[0])
        return

    save_path = os.path.join(SAVE_DIR, data_id + ".h5")
    print(save_path)
    truck_dir = os.path.dirname(save_path)
    if not os.path.exists(truck_dir):
        os.makedirs(truck_dir)

    with h5py.File(save_path, 'w') as fp:
        fp.create_dataset("vec", data=cad_vec, dtype=np.int)

names = os.listdir(name_ROOT)
num= 0
for name in names: 
    data_path = os.path.join(data_root,name[:-4]+'.json')
    #print('data_path: ',data_path)
    # with open(data_path, "r") as fp:
    #     all_data = json.load(fp)
    process_one(data_id=name[:-4],data_path = data_path)
    num += 1
    print(num)
# with open(RECORD_FILE, "r") as fp:
#     all_data = json.load(fp)

# Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data["train"])
# Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data["validation"])
# Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data["test"])
