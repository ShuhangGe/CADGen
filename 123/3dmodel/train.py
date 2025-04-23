
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
from utils.misc import *
import time
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datasets.dataset import CADGENdataset
from model.model_simple import Views2Points
from timm.scheduler import CosineLRScheduler
from model.loss import CADLoss
import os
import os, sys
# online package
# optimizer
import torch.optim as optim
import numpy as np
#from apex import amp
from torch.cuda.amp import autocast as autocast
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# utils




def build_opti_sche(base_model, config):
    opti_config = config.optimizer
    print(opti_config)
    if opti_config['type'] == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    # print(name)
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]
        param_groups = add_weight_decay(base_model, weight_decay=opti_config['kwargs']['weight_decay'])
        optimizer = optim.AdamW(param_groups, **opti_config['kwargs'])
    elif opti_config['type'] == 'Adam':
        optimizer = optim.Adam(base_model.parameters(), **opti_config['kwargs'])
    elif opti_config['type'] == 'SGD':
        optimizer = optim.SGD(base_model.parameters(), nesterov=True, **opti_config['kwargs'])
    else:
        raise NotImplementedError()

    sche_config = config.scheduler
    if sche_config['type'] == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config['kwargs'])  # misc.py
    elif sche_config['type'] == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                t_initial=sche_config['kwargs']['epochs'],
                t_mul=1,
                lr_min=1e-6,
                decay_rate=0.1,
                warmup_lr_init=1e-6,
                warmup_t=sche_config['kwargs']['initial_epochs'],
                cycle_limit=1,
                t_in_epochs=True)
    elif sche_config['type'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config['kwargs'])
    elif sche_config['type'] == 'function':
        scheduler = None
    else:
        raise NotImplementedError()
    
    return optimizer, scheduler
def main():
    # args
    args = parser.get_args()
    # CUDA
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = CADGENdataset(args,test=False)
    test_data = CADGENdataset(args,test=True)
    train_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=args.train_batch,
                                            shuffle=True,
                                            num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size=args.test_batch,
                                               shuffle=True,
                                               num_workers=args.num_workers)
    model = Views2Points(args)
    #print(model)
    optimizer,scheduler = build_opti_sche(model,args)
    
    # optimizer = torch.optim.Adam(model.parameters(),lr=args.lr ,weight_decay=0.05)
    # scheduler = CosineLRScheduler(optimizer,
    #     t_initial=args.epochs,
    #     t_mul=1,
    #     lr_min=1e-6,
    #     decay_rate=0.1,
    #     warmup_lr_init=1e-6,
    #     warmup_t=args.initial_epochs,
    #     cycle_limit=1,
    #     t_in_epochs=True)
    loss_fun = CADLoss(args).to(args.device)
    model = model.to(args.device)
    train_num  = 0
    writer = SummaryWriter(args.log_path)
    best_test = 100000

    for epoch in range(args.epochs):
        loss_cmd_train = 0
        loss_args_train = 0
        train_num = 0

        for index, data in enumerate(train_loader,0):
            model.train()
            front_pic,top_pic,side_pic,cad_data,command,paramaters, data_num = data
            front_pic = front_pic.to(args.device)
            top_pic = top_pic.to(args.device)
            side_pic = side_pic.to(args.device)
            cad_data = cad_data.to(args.device)
            command = command.to(args.device)
            paramaters = paramaters.to(args.device)
            '''cad_data.shape:  torch.Size([50, 1024, 3])'''
            with autocast():
                output = model(front_pic,top_pic,side_pic,cad_data)
                output["tgt_commands"] = command
                output["tgt_args"] = paramaters
                loss_dict = loss_fun(output)
            print('len(train_loader): ',len(train_loader),'index: ',index)
            loss_cmd_train += loss_dict["loss_cmd"]
            loss_args_train += loss_dict["loss_args"]
            train_num += 1
            #print('sum(loss_dict.values()): ',sum(loss_dict.values()))
            #print('loss_dict.values(): ',loss_dict.values())
            loss = sum(loss_dict.values())
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            print('loss: ',loss)
        loss_cmd_train = loss_cmd_train/train_num
        loss_args_train = loss_args_train/train_num     
        writer.add_scalar('loss_cmd_train', loss_cmd_train, global_step=epoch)
        writer.add_scalar('loss_args_train', loss_args_train, global_step=epoch)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        loss_cmd_test = 0
        loss_args_test = 0
        test_num = 0
        with torch.no_grad():
            for index, data in enumerate(test_loader,0):
                front_pic,top_pic,side_pic,cad_data,command,paramaters, data_num = data
                front_pic = front_pic.to(args.device)
                top_pic = top_pic.to(args.device)
                side_pic = side_pic.to(args.device)
                cad_data = cad_data.to(args.device)
                command = command.to(args.device)
                paramaters = paramaters.to(args.device)
                '''cad_data.shape:  torch.Size([50, 1024, 3])'''
                with autocast():
                    output_test = model(front_pic,top_pic,side_pic,cad_data)
                    #print('output: ',output)
                    output_test["tgt_commands"] = command
                    output_test["tgt_args"] = paramaters
                    loss_dict_test = loss_fun(output_test)
                    loss_cmd_test += loss_dict["loss_cmd"]
                    loss_args_test += loss_dict["loss_args"]
                    loss_test = sum(loss_dict_test.values())
                test_num += 1
                print('loss_test: ',loss_test)
        loss_cmd_test = loss_cmd_test / test_num  
        loss_args_test = loss_args_test / test_num    
        writer.add_scalar('loss_cmd_test', loss_cmd_test, global_step=epoch)
        writer.add_scalar('loss_args_test', loss_args_test, global_step=epoch)
        if best_test > loss_cmd_test :
            best_test = loss_args_test
            model_save = os.path.join(args.model_path,f'CADGEN_best')
            torch.save(model.state_dict(), model_save)
        if epoch%5==0:
            model_save = os.path.join(args.model_path,f'CADGEN_{epoch}')
            torch.save(model.state_dict(), model_save)
    model_save = os.path.join(args.model_path,f'CADGEN_latest')
    torch.save(model.state_dict(), model_save)
if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
