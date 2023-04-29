import os
import glob
import json
import numpy as np
import random
import h5py
from joblib import Parallel, delayed
from trimesh.sample import sample_surface
import argparse
import sys
sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.visualize import CADsolid2pc, create_CAD
from utils.pc_utils import write_ply, read_ply

DATA_ROOT = "/home/rl4citygen/gsh/data/deepcad"
RAW_DATA = os.path.join(DATA_ROOT, "Sketch_1_Extrude_1_ply")
#RECORD_FILE = os.path.join(DATA_ROOT, "train_val_test_split.json")
RECORD_FILE = '/home/rl4citygen/gsh/DeepCAD/DeepCAD-master/data/splited_json/Sketch_1_Extrude_1'
N_POINTS = 10000 # 4096
WRITE_NORMAL = False
SAVE_DIR = os.path.join(DATA_ROOT, "pc_cad")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

INVALID_IDS = []

def process_one(data_id,data_path,NAME):

    if data_id in INVALID_IDS:
        print("skip {}: in invalid id list".format(data_id))
        return

    save_path = os.path.join(SAVE_DIR, NAME[:-5] + ".ply")


    json_path = data_path
    with open(json_path, "r") as fp:
        data = json.load(fp)

    try:
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        shape = create_CAD(cad_seq)
    except Exception as e:
        print("create_CAD failed:", data_id)
        return None

    try:
        out_pc = CADsolid2pc(shape, N_POINTS, int(data_id))#.split("/")[-1])
    except Exception as e:
        print("convert point cloud failed:", data_id)
        return None

    save_path = os.path.join(SAVE_DIR, NAME[:-5] + ".ply")
    truck_dir = os.path.dirname(save_path)
    if not os.path.exists(truck_dir):
        os.makedirs(truck_dir)

    write_ply(out_pc, save_path)


# with open(RECORD_FILE, "r") as fp:
#     all_data = json.load(fp)
names = os.listdir(RECORD_FILE)
DATA_ID = 0
for name in names:
    file_path = os.path.join(RECORD_FILE,name)
    process_one(data_id=DATA_ID,data_path = file_path,NAME=name)
    print('DATA_ID: ',DATA_ID)
    DATA_ID += 1
# process_one(all_data["train"][3])
# exit()

# parser = argparse.ArgumentParser()
# parser.add_argument('--only_test', action="store_true", help="only convert test data")
# args = parser.parse_args()
# if not args.only_test:
#     Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data)#["train"])
#     Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data["validation"])
# Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data["test"])
