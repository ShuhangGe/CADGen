
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

def collate_fn(batch):
    '''
     collate_fn (callable, optional): merges a list of samples to form a mini-batch.
     该函数参考touch的default_collate函数，也是DataLoader的默认的校对方法，当batch中含有None等数据时，
     默认的default_collate校队方法会出现错误
     一种的解决方法是：
     判断batch中image是否为None,如果为None，则在原来的batch中清除掉，这样就可以在迭代中避免出错了
    :param batch:
    :return:
    '''
    r"""Puts each data field into a tensor with outer dimension batch size"""
    # 这里添加：判断image是否为None,如果为None，则在原来的batch中清除掉，这样就可以在迭代中避免出错了
    if isinstance(batch, list):
        batch = [(image, image_id) for (image, image_id) in batch if image is not None]
    if batch==[]:
        return (None,None)
 
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(error_msg_fmt.format(elem.dtype))
 
            return collate_fn([torch.from_numpy(b) for b in batch])
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(batch[0], int_classes):
        return torch.tensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):  # namedtuple
        return type(batch[0])(*(collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)#ok
        return [collate_fn(samples) for samples in transposed]
 
    raise TypeError((error_msg_fmt.format(type(batch[0]))))


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
                                            shuffle=False,
                                            num_workers=args.num_workers)
    #print('2222222222222222222222222222222222222222222222222222222')
    test_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size=args.test_batch,
                                               shuffle=False,
                                               num_workers=args.num_workers)
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
        #print('5555555555555555555555555555555555555555555555555')

        for index, data in enumerate(train_loader,0):
            #print('000000000000000000000000000000000')
            print('data_id: ', data['id'])
            model.train()
            front_pic,top_pic,side_pic,cad_data,command,paramaters = data['data']
            front_pic = front_pic.to(args.device)
            top_pic = top_pic.to(args.device)
            side_pic = side_pic.to(args.device)
            cad_data = cad_data.to(args.device)
            command = command.to(args.device)
            paramaters = paramaters.to(args.device)
            '''cad_data.shape:  torch.Size([50, 1024, 3])'''
            #print('5555555555555555555555555555555555555555')
            with autocast():
                output = model(front_pic,top_pic,side_pic,cad_data,command,paramaters)
                #print('output: ',output)
                #print('6666666666666666666666666666666666666666')
                output["tgt_commands"] = command
                output["tgt_args"] = paramaters
                loss_dict = loss_fun(output)
            # with autocast():
                
            print('len(train_loader): ',len(train_loader),'index: ',index)
            loss_cmd_train += loss_dict["loss_cmd"]
            loss_args_train += loss_dict["loss_args"]
            train_num += 1
            #print('sum(loss_dict.values()): ',sum(loss_dict.values()))
            #print('loss_dict.values(): ',loss_dict.values())
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            print('loss: ',loss)
            # with torch.autograd.detect_anomaly():
            loss.backward()
            
            #logname = '/scratch/sg7484/CMDGen/results/paramaters' + f'/log_{index}.log'
            #log = Logger(logname,level='debug')

            for name, param in model.named_parameters():
                #log.logger.info(f'-->name: {name}, -->grad_requirs: {param.requires_grad},-->grad_value: {param.grad},\n')
                if not torch.isfinite(param.grad).all():
                    print(name, torch.isfinite(param.grad).all())
            #torch.save(model.state_dict(), f'/scratch/sg7484/CMDGen/results/paramaters/paramater_{index}')
            optimizer.step()
            #torch.save(model.state_dict(), f'/scratch/sg7484/CMDGen/results/paramaters/paramater_after_{index}')
            
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
                front_pic,top_pic,side_pic,cad_data,command,paramaters = data['data']
                front_pic = front_pic.to(args.device)
                top_pic = top_pic.to(args.device)
                side_pic = side_pic.to(args.device)
                cad_data = cad_data.to(args.device)
                command = command.to(args.device)
                paramaters = paramaters.to(args.device)
                '''cad_data.shape:  torch.Size([50, 1024, 3])'''
                print('---------------')
                with autocast():
                    output_test = model(front_pic,top_pic,side_pic,cad_data)
                    #print('output: ',output)
                    output_test["tgt_commands"] = command
                    output_test["tgt_args"] = paramaters

                    loss_dict_test = loss_fun(output_test)
                    loss_cmd_test += loss_dict["loss_cmd"]
                    loss_args_test += loss_dict["loss_args"]
                    loss_test = sum(loss_dict_test.values())
                #with autocast():
                    
                    
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
        if epoch%50==0:
            model_save = os.path.join(args.model_path,f'CADGEN_{epoch}')
            torch.save(model.state_dict(), model_save)
    model_save = os.path.join(args.model_path,f'CADGEN_latest')
    torch.save(model.state_dict(), model_save)
if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
