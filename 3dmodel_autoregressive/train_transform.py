
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
from utils.misc import *
import time
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datasets.dataset import CADGENdataset
from datasets.data_collate import collate_fn
from model.model_transformer import Views2Points
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
import logging
from logging import handlers

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# utils



class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        # sh = logging.StreamHandler()#往屏幕上输出
        # sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        #self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)


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
    
    # if config.get('bnmscheduler') is not None:
    #     bnsche_config = config.bnmscheduler
    #     if bnsche_config.type == 'Lambda':
    #         bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
    #     scheduler = [scheduler, bnscheduler]
    
    return optimizer, scheduler
def main():
    # args
    args = parser.get_args()
    #print('args: ',args)
    #print('000000000000000000000000000000000')
    # CUDA
    if args.device == 'GPU':
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device('cpu')
    train_data = CADGENdataset(args,test=False)
    #print('1111111111111111111111111111111111111111111')
    test_data = CADGENdataset(args,test=True)
    train_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=args.train_batch,
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            collate_fn=collate_fn)
    #print('2222222222222222222222222222222222222222222222222222222')
    test_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size=args.test_batch,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               collate_fn=collate_fn)
    #print('33333333333333333333333333333333333333333333333333')
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
    #print('44444444444444444444444444444444444444444444444')
    for epoch in range(args.epochs):
        loss_cmd_train = 0
        loss_args_train = 0
        train_num = 0
        loss_sum = 0
        #print('5555555555555555555555555555555555555555555555555')
        for index, data in enumerate(train_loader,0):
            #print('000000000000000000000000000000000')
            
            model.train()
            front_pic,top_pic,side_pic,cad_data,command,paramaters,data_num= data
            print('data_id: ', data_num)
            front_pic = front_pic.to(args.device)
            top_pic = top_pic.to(args.device)
            side_pic = side_pic.to(args.device)
            cad_data = cad_data.to(args.device)
            command = command.to(args.device)
            paramaters = paramaters.to(args.device)
            train_command = command[:,:-1]
            train_paramaters = paramaters[:,:-1,:]
            tgt_commands = command[:,1:]
            #print('tgt_commands.shape: ',tgt_commands.shape)
            tgt_paramaters = paramaters[:,1:,:]
            #print('tgt_paramaters.shape: ',tgt_paramaters.shape)
            # print('train_command.shape: ',train_command.shape)
            # print('train_paramaters.shape: ',train_paramaters.shape)
            # print('tgt_commands.shape: ',tgt_commands.shape)
            # print('tgt_paramaters.shape: ',tgt_paramaters.shape)
            # print('train_command: ',train_command)
            # print('tgt_commands: ',tgt_commands)
            '''cad_data.shape:  torch.Size([50, 1024, 3])'''
            #print('5555555555555555555555555555555555555555')
            with autocast():
                output = model(front_pic,top_pic,side_pic,cad_data,train_command,train_paramaters)
                #print('output: ',output)
                #print('6666666666666666666666666666666666666666')
                output["tgt_commands"] = tgt_commands
                output["tgt_args"] = tgt_paramaters
                loss_dict = loss_fun(output)
            # with autocast():
                
            print('epoch: ',epoch, 'len(train_loader): ',len(train_loader),'index: ',index)
            loss_cmd_train += loss_dict["loss_cmd"]
            loss_args_train += loss_dict["loss_args"]
            train_num += 1
            #print('sum(loss_dict.values()): ',sum(loss_dict.values()))
            #print('loss_dict.values(): ',loss_dict.values())
            loss = sum(loss_dict.values())
            print('loss: ',loss)
            loss_sum = loss_sum +loss
            print('loss_cmd_train: ',loss_dict["loss_cmd"], 'loss_args_train: ',loss_dict["loss_args"])
            assert torch.isnan(loss).sum() == 0, print('loss: ',loss)
            
            optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=20, norm_type=2)

            #logname = '/scratch/sg7484/CMDGen/results/paramaters' + f'/log_{index}.log'
            #log = Logger(logname,level='debug')

            # for name, param in model.named_parameters():
            #     #log.logger.info(f'-->name: {name}, -->grad_requirs: {param.requires_grad},-->grad_value: {param.grad},\n')
            #     if not torch.isfinite(param.grad).all():
            #         print(name, torch.isfinite(param.grad).all())
            
            #torch.save(model.state_dict(), f'/scratch/sg7484/CMDGen/results/paramaters/paramater_{index}')
            optimizer.step()
            #torch.save(model.state_dict(), f'/scratch/sg7484/CMDGen/results/paramaters/paramater_after_{index}')
       
        loss_cmd_train = loss_cmd_train/train_num
        loss_args_train = loss_args_train/train_num
        loss_sum = loss_sum/train_num     
        print('loss_train_sum: ',loss_sum)
        writer.add_scalar(f'loss_cmd_train_{args.lr}', loss_cmd_train, global_step=epoch)
        writer.add_scalar(f'loss_args_train_{args.lr}', loss_args_train, global_step=epoch)
        writer.add_scalar(f'loss_sum_{args.lr}', loss_sum, global_step=epoch)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        loss_cmd_test = 0
        loss_args_test = 0
        test_num = 1e-9
        loss_test_sum = 0
        with torch.no_grad():
            for index, data in enumerate(test_loader,0):
                front_pic,top_pic,side_pic,cad_data,command,paramaters,data_num = data
                print('data_id: ', data_num)

                front_pic = front_pic.to(args.device)
                top_pic = top_pic.to(args.device)
                side_pic = side_pic.to(args.device)
                cad_data = cad_data.to(args.device)
                command = command.to(args.device)
                paramaters = paramaters.to(args.device)
                train_command = command[:,:-1]
                train_paramaters = paramaters[:,:-1,:]
                tgt_commands = command[:,1:]
                tgt_paramaters = paramaters[:,1:,:]
                '''cad_data.shape:  torch.Size([50, 1024, 3])'''
                #print('---------------')
                with autocast():
                    output_test = model(front_pic,top_pic,side_pic,cad_data,train_command,train_paramaters)
                    #print('output: ',output)
                    output_test["tgt_commands"] = tgt_commands
                    output_test["tgt_args"] = tgt_paramaters

                    loss_dict_test = loss_fun(output_test)
                    loss_cmd_test += loss_dict["loss_cmd"]
                    loss_args_test += loss_dict["loss_args"]
                    loss_test = sum(loss_dict_test.values())
                #with autocast():
                loss_test_sum += loss_test
                    
                test_num += 1
                assert torch.isnan(loss_test).sum() == 0, print('loss_test: ',loss_test)
        loss_cmd_test = loss_cmd_test / test_num
        loss_args_test = loss_args_test / test_num
  
        loss_test_sum = loss_test_sum / test_num    
        writer.add_scalar(f'loss_cmd_test_{args.lr}', loss_cmd_test, global_step=epoch)
        writer.add_scalar(f'loss_args_test_{args.lr}', loss_args_test, global_step=epoch)
        writer.add_scalar(f'loss_test_sum_{args.lr}', loss_test_sum, global_step=epoch)
        print('loss_test_sum: ',loss_test_sum)
        if best_test > loss_cmd_test :
            best_test = loss_args_test
            model_save = os.path.join(args.model_path,f'CADGEN_best_train_{loss_sum}_test_{loss_test_sum}')
            torch.save(model.state_dict(), model_save)
        if epoch%1==0:
            model_save = os.path.join(args.model_path,f'CADGEN_{epoch}_train_{loss_sum}_test_{loss_test_sum}')
            torch.save(model.state_dict(), model_save)
    model_save = os.path.join(args.model_path,f'CADGEN_latest_train_{loss_sum}_test_{loss_test_sum}')
    torch.save(model.state_dict(), model_save)
if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
