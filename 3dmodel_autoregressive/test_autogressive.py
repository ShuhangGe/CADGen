from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
from utils.misc import *
import time
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datasets.dataset import CADGENdataset
from model.model import Views2Points
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
torch.set_printoptions(profile="full")

from cadlib.macro import *

def logits2vec( outputs, refill_pad=True, to_numpy=True):
    """network outputs (logits) to final CAD vector"""
    #print('outputs[command_logits]: ', outputs['command_logits'])
    #print('outputs[args_logits]: ', outputs['args_logits'])
    out_command = torch.argmax(torch.softmax(outputs['command_logits'], dim=-1), dim=-1)  # (N, S)
    out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1  # (N, S, N_ARGS)
    if refill_pad: # fill all unused element to -1
        mask = ~torch.tensor(CMD_ARGS_MASK).bool().cuda()[out_command.long()]
        out_args[mask] = -1

    out_cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)
    if to_numpy:
        out_cad_vec = out_cad_vec.detach().cpu().numpy()
    return out_cad_vec

def main():
    cfg = parser.get_args()
    max_len = MAX_TOTAL_LEN
    
    
    model = Views2Points(cfg)
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load from checkpoint if provided
    model_path = '/scratch/sg7484/CMDGen/results/train_1e-7_fulldata_autoregressive/model/CADGEN_5'
    model_par = torch.load(model_path)
    model.load_state_dict(model_par)
    model = model.to(cfg.device)
    model.eval()

    # create dataloader
    dataset_test = CADGENdataset(cfg, test=True)
    test_loader = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=cfg.num_workers)
    print("Total number of test data:", len(test_loader))

    if not os.path.exists(cfg.test_outputs):
        os.makedirs(cfg.test_outputs)
    #cfg.outputs = "{}/results/test_{}".format(cfg.exp_dir, cfg.ckpt)


    # evaluate

    for i, data in enumerate(test_loader):
        # if i ==0:
        #     continue
        front_pic,top_pic,side_pic,cad_data,command,paramaters, data_num= data
        print('data_id: ', data_num)

        #print('command.shape: ',command.shape)
        #print('paramaters.shape: ',paramaters.shape)
        front_pic = front_pic.to(cfg.device)
        top_pic = top_pic.to(cfg.device)
        side_pic = side_pic.to(cfg.device)
        cad_data = cad_data.to(cfg.device)
        command = command.to(cfg.device)
        paramaters = paramaters.to(cfg.device)
        # print('command[:,:2,:]: ',command[:,:2])
        # print('paramaters[:,:2,:]: ',paramaters[:,:2,:])
        tgt_commands = command[:,1:]
        tgt_paramaters = paramaters[:,1:,:]
        
        # command_ys = command[:,:2]
        # paramaters_ys = paramaters[:,:2,:]
        #print('command_ys: ',command_ys)
        gt_vec = torch.cat([command.unsqueeze(-1), paramaters], dim=-1).squeeze(1).detach().cpu().numpy()
        command_ys = args = torch.ones(1, 1).fill_(ALL_COMMANDS.index('SOL')). \
            type(torch.long).to(cfg.device)
        #print('command_ys.shape: ',command_ys.shape)
        '''command_ys.shape:  torch.Size([1, 1])'''
        paramaters_ys = torch.ones(1, 1, N_ARGS).fill_(PAD_VAL). \
            type(torch.long).to(cfg.device)
        #print('paramaters_ys.shape: ',paramaters_ys.shape)
        '''paramaters_ys.shape:  torch.Size([1, 1, 16])'''
            
        # cad_vec = torch.cat([paramaters_ys.unsqueeze(0),paramaters_ys],dim=-1)
        # pad_len = max_len - cad_vec.shape[0]
        # cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        # command_ys = cad_vec[:, 0]
        # paramaters_ys = cad_vec[:, 1:]
        #print('command_ys: ',command_ys.shape)
        #print('paramaters_ys: ',paramaters_ys.shape)
        #print('gt_vec.shape: ',gt_vec.shape)
        #commands_ = gt_vec[:, :, 0]
        z = model.forward_encoder(front_pic,top_pic,side_pic,cad_data,command_ys, paramaters_ys)
        #print('z.shape: ',z.shape)
        '''z.shape:  torch.Size([1, 64, 256])'''
        for j in range(max_len-1):
            with torch.no_grad():
                with autocast():
                    #print('start model')
                    outputs = model.forward_decoder(z,command_ys, paramaters_ys)
                    # command_ys = outputs['command_logits']
                    # paramaters_ys = outputs['args_logits']

                    outputs["command_logits"]= outputs["command_logits"][:,-1,:]
                    #print('outputs["command_logits"]: ',outputs["command_logits"])
                    outputs["args_logits"]= outputs["args_logits"][:,-1,:,:]
                    #print('outputs["command_logits"].shape: ',outputs["command_logits"].shape,'outputs["args_logits"].shape: ',outputs["args_logits"].shape)
                    '''outputs["command_logits"].shape:  torch.Size([1, 6]) outputs["args_logits"].shape:  torch.Size([1, 16, 257])'''
                    out_cad_vec = logits2vec(outputs)
                    out_cad_vec = out_cad_vec[None,:,:]
                    #print('out_cad_vec.shape: ',out_cad_vec.shape)
                    '''out_cad_vec.shape:  (1, 1, 17)'''
                    # print('out_cad_vec.shape: ',out_cad_vec.shape)                    
                    command_temp = torch.tensor(out_cad_vec[:,-1,0])
                    # print('command_temp: ',command_temp)
                    #print('command_temp.shape: ',command_temp.shape)
                    command_temp = command_temp.to(cfg.device)
                    paramaters_temp = torch.tensor(out_cad_vec[:,-1,1:])
                    paramaters_temp = paramaters_temp.to(cfg.device)

                    #print('paramaters_temp: ',paramaters_temp)
                    # print('paramaters_temp.shape: ',paramaters_temp.shape)

                    command_ys = torch.cat([command_ys,command_temp.unsqueeze(0)],dim=-1)
                    paramaters_ys = torch.cat([paramaters_ys,paramaters_temp.unsqueeze(0)],dim = 1)
                    #print(f'command_ys.shape: {command_ys.shape}, paramaters_ys.shape: {paramaters_ys.shape}')
                    '''command_ys.shape: torch.Size([1, 2]), paramaters_ys.shape: torch.Size([1, 2, 16])'''
                    #print('command_ys.shape: ',command_ys.shape)
                    #print('paramaters_ys.shape: ',paramaters_ys.shape)
                    
                    
                    # if command_temp[0] == ALL_COMMANDS.index('EOS'):
                    #     break
        print('tgt_commands: ',tgt_commands)
        print('command_ys: ',command_ys)
        ##print('tgt_paramaters: ',tgt_paramaters)
        #print('paramaters_ys: ',paramaters_ys)
        
        #break
                    #torch.cat(command_ys,out_cad_vec)
#                     '''out_logits[0].shape:  torch.Size([1, 64, 6])
#                     out_logits[1].shape:  torch.Size([1, 64, 16, 257])'''
#                     outputs["tgt_commands"] = command
#                     outputs["tgt_args"] = paramaters

                    
#                     batch_out_vec = logits2vec(outputs)
#         for j in range(batch_size):
#             out_vec = batch_out_vec[j]
#             seq_len = commands_[j].tolist().index(EOS_IDX)
#             print('data["id"]: ',data["id"])
#             data_id = data["id"][j].split('/')[-1]

#             save_path = os.path.join(cfg.test_outputs, '{}_vec.h5'.format(data_id))
#             print('save_path: ',save_path)
#             with h5py.File(save_path, 'w') as fp:
#                 fp.create_dataset('out_vec', data=out_vec[:seq_len], dtype=np.int)
#                 fp.create_dataset('gt_vec', data=gt_vec[j][:seq_len], dtype=np.int)

if __name__=='__main__':
    # side = torch.full((1,3,200,200),255).to(device)  
    # front = torch.full((1,3,200,200),255).to(device)
    # top = torch.full((1,3,200,200),255).to(device)
    # cad_data = torch.rand(1, 1024, 3).to(device)
    main()
    
    
    
#padding
'''tgt_commands:  tensor([[0, 0, 0, 0, 4, 2, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]], device='cuda:0')
command_ys:  tensor([[4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0')'''
         
