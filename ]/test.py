
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datasets.Dataset import Point2CAD_dataset
from model.model import Point_MAE
from timm.scheduler import CosineLRScheduler
from model.loss import CADLoss
import os
import os, sys
# online package
# optimizer
import torch.optim as optim
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# utils
from utils.logger import *
from utils.misc import *
from timm.scheduler import CosineLRScheduler
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
model = Point_MAE(cfg)
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load from checkpoint if provided
model_path = '/home/shuhang/Desktop/CADgen/models/CADGEN_best'
model_par = torch.load(model_path)
model.load_state_dict(model_par)
model = model.to(cfg.device)
model.eval()

# create dataloader
dataset_test = Point2CAD_dataset(cfg, 'test')
test_loader = torch.utils.data.DataLoader(dataset_test,
                                            batch_size=cfg.test_batch,
                                            shuffle=True,
                                            num_workers=cfg.num_workers)
print("Total number of test data:", len(test_loader))

if not os.path.exists(cfg.test_outputs):
    os.makedirs(cfg.test_outputs)
#cfg.outputs = "{}/results/test_{}".format(cfg.exp_dir, cfg.ckpt)


# evaluate

for i, data in enumerate(test_loader):
    command, paramaters, cad_data = data['data']
    batch_size = command.shape[0]
    # commands = data['command']
    # args = data['args']
    commands = command.to(cfg.device)
    args = paramaters.to(cfg.device)
    cad_data = cad_data.to(cfg.device)
    #print('commands.shape:',commands.shape)
    gt_vec = torch.cat([commands.unsqueeze(-1), args], dim=-1).squeeze(1).detach().cpu().numpy()
    #print('gt_vec.shape: ',gt_vec.shape)
    commands_ = gt_vec[:, :, 0]
    with torch.no_grad():
        outputs= model(cad_data)
        print(outputs.keys())
        batch_out_vec = logits2vec(outputs)

    for j in range(batch_size):
        out_vec = batch_out_vec[j]
        seq_len = commands_[j].tolist().index(EOS_IDX)
        print('data["id"]: ',data["id"])
        data_id = data["id"][j].split('/')[-1]

        save_path = os.path.join(cfg.test_outputs, '{}_vec.h5'.format(data_id))
        print('save_path: ',save_path)
        with h5py.File(save_path, 'w') as fp:
            fp.create_dataset('out_vec', data=out_vec[:seq_len], dtype=np.int)
            fp.create_dataset('gt_vec', data=gt_vec[j][:seq_len], dtype=np.int)
