from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
from utils.misc import *
import time
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datasets.dataset import CADGENdataset
from model.model_deformerable import Views2Points
from timm.scheduler import CosineLRScheduler
from model.loss import CADLoss
import os
import os, sys
# online package
# optimizer
import torch.optim as optim
import numpy as np
from torch.cuda.amp import autocast as autocast
import h5py

from cadlib.macro import *

def logits2vec( outputs, refill_pad=True, to_numpy=True):
    """network outputs (logits) to final CAD vector"""
    out_command = torch.argmax(torch.softmax(outputs['command_logits'], dim=-1), dim=-1)  # (N, S)
    out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1  # (N, S, N_ARGS)
    if refill_pad: # fill all unused element to -1
        mask = ~torch.tensor(CMD_ARGS_MASK).bool().cuda()[out_command.long()]
        out_args[mask] = -1

    out_cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)
    if to_numpy:
        out_cad_vec = out_cad_vec.detach().cpu().numpy()
    return out_cad_vec


cfg = parser.get_args()
model = Views2Points(cfg)
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load from checkpoint if provided
model_path = '/scratch/sg7484/CMDGen/results/noauto/fulldata_noauto_deformable3_res18_unet4_lrtest_retrainable/fulldata_noauto_deformable3_res18_unet4_1e-4/model/CADGEN_best_35_4.137737274169922_test_4.850296974182129.path'
model_par = torch.load(model_path)
model.load_state_dict(model_par['model_dict'])
model = model.to(cfg.device)
#model.eval()

# create dataloader
dataset_test = CADGENdataset(cfg, test =True)
test_loader = torch.utils.data.DataLoader(dataset_test,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=cfg.num_workers)
print("Total number of test data:", len(test_loader))

if not os.path.exists(cfg.test_outputs):
    os.makedirs(cfg.test_outputs)
#cfg.outputs = "{}/results/test_{}".format(cfg.exp_dir, cfg.ckpt)
cfg.test_outputs = '/scratch/sg7484/CMDGen/results/noauto/output/fulldata_noauto_deformable3_res18_unet4_best_35_4.13_test_4.85_1e-4_test'
if not os.path.exists(cfg.test_outputs):
    os.makedirs(cfg.test_outputs)
# evaluate
gen_num = 0
total_gen = 100
for i, data in enumerate(test_loader):
    front_pic,top_pic,side_pic,cad_data,command,paramaters , data_num = data
    front_pic = front_pic.to(cfg.device)
    top_pic = top_pic.to(cfg.device)
    side_pic = side_pic.to(cfg.device)
    cad_data = cad_data.to(cfg.device)
    command = command.to(cfg.device)
    paramaters = paramaters.to(cfg.device)
    batch_size = command.shape[0]
    #print('commands.shape:',commands.shape)
    gt_vec = torch.cat([command.unsqueeze(-1), paramaters], dim=-1).squeeze(1).detach().cpu().numpy()
    #print('gt_vec.shape: ',gt_vec.shape)
    commands_ = gt_vec[:, :, 0]
    with torch.no_grad():
        with autocast():
            outputs = model(front_pic,top_pic,side_pic,cad_data)
            #print('output: ',output)
            #print('6666666666666666666666666666666666666666')
            outputs["tgt_commands"] = command
            outputs["tgt_args"] = paramaters
            batch_out_vec = logits2vec(outputs)
    #print('batch_out_vec: ',batch_out_vec)
    for j in range(batch_size):
        out_vec = batch_out_vec[j]
        seq_len = commands_[j].tolist().index(EOS_IDX)

        print('data_num: ',data_num)
        data_idex = int(data_num[0])
        save_path = os.path.join(cfg.test_outputs, '{}_vec.h5'.format(data_idex))
        print('save_path: ',save_path)
        with h5py.File(save_path, 'w') as fp:
            fp.create_dataset('out_vec', data=out_vec[:seq_len], dtype=np.int)
            fp.create_dataset('gt_vec', data=gt_vec[j][:seq_len], dtype=np.int)
        gen_num+=1
        if gen_num>=total_gen:
            break
    if gen_num>=total_gen:
        break
