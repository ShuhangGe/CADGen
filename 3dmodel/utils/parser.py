import os
import argparse
from pathlib import Path
from cadlib.macro import *
DATA_ROOT = '/scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1_test'
CAD_ROOT = '/scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1_test/cad'
CMD_ROOT = '/scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1_test/cmd'
PIC_ROOT = '/scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1_test/pic'
def get_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='CADGen')
    #data_paramaters
    parser.add_argument('--data_root', type=str, default=DATA_ROOT)#
    parser.add_argument('--cad_root', type=str, default=CAD_ROOT)#
    parser.add_argument('--cmd_root', type=str, default=CMD_ROOT)#
    parser.add_argument('--pic_root', type=str, default=PIC_ROOT)#
    
    #model
    parser.add_argument('--resnet_in', type=int, default=16)
    parser.add_argument('--resnet_out', type=int, default=32)
    parser.add_argument('--UNet_out', type=int, default=256)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='GPU')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--initial_epochs', type=int, default=10)
    parser.add_argument('--train_batch', type=int, default=50)
    parser.add_argument('--test_batch', type=int, default=10)
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--save_path', type=str, default='../results/exp1')
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--retrain_model', type=str)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--n_points', type=int, default=10000)
    parser.add_argument('--npoints', type=int, default=1536)
    parser.add_argument('--grid_sample', type=int, default=8)# sqt(npoints,3)
    parser.add_argument('--optimizer', type=dict, default={ 'type': 'AdamW',
                                                    'kwargs': {
                                                    'lr' : 0.001,
                                                    'weight_decay' : 0.05
                                                    }})
    parser.add_argument('--scheduler', type=dict, default={'type': 'CosLR',
                                                            'kwargs': {
                                                                'epochs': 300,
                                                                'initial_epochs' : 10
                                                            }})

    #test
    parser.add_argument('--test_outputs', type=str, default='../results/commands_out')
    
    #encoder paramaters
    parser.add_argument('--NAME', type=str, default='Point_MAE')
    parser.add_argument('--group_size', type=int, default=3)
    #group_size^num_group = npoints
    parser.add_argument('--mask_ratio', type=float, default=0.6)
    parser.add_argument('--mask_type', type=str, default='rand')
    parser.add_argument('--num_group', type=int, default=512)#64
    parser.add_argument('--num_heads', type=int, default=4)#6
    parser.add_argument('--trans_dim', type=int, default=256)#384
    '''num_group*num_heads = trans_dim'''
    parser.add_argument('--encoder_dims', type=int, default=256)#384
    parser.add_argument('--depth', type=int, default=6)#12encoder depth
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--decoder_depth', type=int, default=4)
    parser.add_argument('--decoder_num_heads', type=int, default=6)

    #decoder paramaters
    parser.add_argument('--args_dim', type=int, default=256)
    parser.add_argument('--n_args', type=int, default=N_ARGS)
    parser.add_argument('--n_commands', type=int, default=len(ALL_COMMANDS))
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_layers_decode', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1 )
    parser.add_argument('--dim_z', type=int, default=256)
    parser.add_argument('--use_group_emb', type=bool, default=True)
    parser.add_argument('--max_n_ext', type=int, default=MAX_N_EXT)
    parser.add_argument('--max_n_loops', type=int, default=MAX_N_LOOPS)
    parser.add_argument('--max_n_curves', type=int, default=MAX_N_CURVES)
    parser.add_argument('--max_num_groups', type=int, default=30)
    parser.add_argument('--max_total_len', type=int, default=MAX_TOTAL_LEN)

    #transformer decoder
    parser.add_argument('--d_model_transdecoder', type=int, default=512)
    parser.add_argument('--nhead_transdecoder', type=int, default=8)
    parser.add_argument('--num_encoder_layers_transdecoder', type=int, default=6)
    parser.add_argument('--num_decoder_layers_transdecoder', type=int, default=6)
    parser.add_argument('--dim_feedforward_transdecoder', type=int, default=2048)
    parser.add_argument('--dropout_transdecoder', type=int, default=0.1)

    #deformable decoder
    parser.add_argument('--d_model_deformable', type=int, default=256)
    parser.add_argument('--nhead_deformable', type=int, default=8)
    parser.add_argument('--k_deformabel', type=int, default=3)
    parser.add_argument('--scales_deformable', type=int, default=3)
    parser.add_argument('--last_point_num', type=int, default=64)
    parser.add_argument('--dim_feedforward_deformable', type=int, default=2048)
    parser.add_argument('--num_decoder_layers_deformable', type=int, default=8)
    parser.add_argument('--num_queries', type=int, default=64)
    #loss paramaters
    parser.add_argument('--loss_weights', type=dict, default={
            "loss_cmd_weight": 1.0,
            "loss_args_weight": 2.0
        })
    args = parser.parse_args()
    save_path = args.save_path
    log_path = os.path.join(save_path,'log')
    model_path = os.path.join(save_path,'model')
    args.log_path = log_path
    args.model_path = model_path
    exp_dir = [args.model_path,args.log_path]
    create_experiment_dir(exp_dir)
    return args

def create_experiment_dir(pathes):
    for path in pathes:
        if not os.path.exists(path):
            os.makedirs(path)
            print('Create experiment path successfully at %s' % path)
            
            
            
            
            
            
# def get_args():
#     parser = argparse.ArgumentParser()
#     parser = argparse.ArgumentParser(description='CADGen')
#     #data_paramaters
#     parser.add_argument('--data_root', type=str, default=DATA_ROOT)
#     parser.add_argument('--cad_root', type=str, default=CAD_ROOT)
#     parser.add_argument('--cmd_root', type=str, default=CMD_ROOT)
    
#     #model
#     parser.add_argument('--resnet_in', type=int, default=16)
#     parser.add_argument('--resnet_out', type=int, default=32)
#     parser.add_argument('--UNet_out', type=int, default=3)

#     parser.add_argument('--lr', type=float, default=1e-4)
#     parser.add_argument('--weight_decay', type=float, default=0.05)
#     parser.add_argument('--device', type=str, default='GPU')
#     parser.add_argument('--epochs', type=int, default=300)
#     parser.add_argument('--initial_epochs', type=int, default=10)
#     parser.add_argument('--train_batch', type=int, default=50)
#     parser.add_argument('--test_batch', type=int, default=10)
#     parser.add_argument('--data_path', type=str, default='../data')
#     parser.add_argument('--save_path', type=str, default='../results/exp1')
#     parser.add_argument('--log_path', type=str)
#     parser.add_argument('--model_path', type=str)
#     parser.add_argument('--num_workers', type=int, default=1)
#     parser.add_argument('--n_points', type=int, default=10000)
#     parser.add_argument('--npoints', type=int, default=1000)
#     parser.add_argument('--grid_sample', type=int, default=10)# sqt(npoints,3)
#     parser.add_argument('--optimizer', type=dict, default={ 'type': 'AdamW',
#                                                     'kwargs': {
#                                                     'lr' : 0.001,
#                                                     'weight_decay' : 0.05
#                                                     }})
#     parser.add_argument('--scheduler', type=dict, default={'type': 'CosLR',
#                                                             'kwargs': {
#                                                                 'epochs': 300,
#                                                                 'initial_epochs' : 10
#                                                             }})

#     #test
#     parser.add_argument('--test_outputs', type=str, default='../results/out')
    
#     #encoder paramaters
#     parser.add_argument('--NAME', type=str, default='Point_MAE')
#     parser.add_argument('--group_size', type=int, default=25)
#     #group_size^num_group = npoints
#     parser.add_argument('--mask_ratio', type=float, default=0.6)
#     parser.add_argument('--mask_type', type=str, default='rand')
#     parser.add_argument('--num_group', type=int, default=40)#64
#     parser.add_argument('--num_heads', type=int, default=4)#6
#     parser.add_argument('--trans_dim', type=int, default=160)#384
#     '''num_group*num_heads = trans_dim'''
#     parser.add_argument('--encoder_dims', type=int, default=160)#384
#     parser.add_argument('--depth', type=int, default=12)
#     parser.add_argument('--drop_path_rate', type=float, default=0.1)

#     parser.add_argument('--decoder_depth', type=int, default=4)
#     parser.add_argument('--decoder_num_heads', type=int, default=6)

#     #decoder paramaters
#     parser.add_argument('--args_dim', type=int, default=256)
#     parser.add_argument('--n_args', type=int, default=N_ARGS)
#     parser.add_argument('--n_commands', type=int, default=len(ALL_COMMANDS))
#     parser.add_argument('--n_layers', type=int, default=4)
#     parser.add_argument('--n_layers_decode', type=int, default=4)
#     parser.add_argument('--n_heads', type=int, default=8)
#     parser.add_argument('--dim_feedforward', type=int, default=512)
#     parser.add_argument('--d_model', type=int, default=256)
#     parser.add_argument('--dropout', type=float, default=0.1 )
#     parser.add_argument('--dim_z', type=int, default=256)
#     parser.add_argument('--use_group_emb', type=bool, default=True)
#     parser.add_argument('--max_n_ext', type=int, default=MAX_N_EXT)
#     parser.add_argument('--max_n_loops', type=int, default=MAX_N_LOOPS)
#     parser.add_argument('--max_n_curves', type=int, default=MAX_N_CURVES)
#     parser.add_argument('--max_num_groups', type=int, default=30)
#     parser.add_argument('--max_total_len', type=int, default=MAX_TOTAL_LEN)

#     #loss paramaters
#     parser.add_argument('--loss_weights', type=dict, default={
#             "loss_cmd_weight": 1.0,
#             "loss_args_weight": 2.0
#         })
#     args = parser.parse_args()
#     save_path = args.save_path
#     log_path = os.path.join(save_path,'log')
#     model_path = os.path.join(save_path,'model')
#     args.log_path = log_path
#     args.model_path = model_path
#     exp_dir = [args.model_path,args.log_path]
#     create_experiment_dir(exp_dir)
#     return args

# def create_experiment_dir(pathes):
#     for path in pathes:
#         if not os.path.exists(path):
#             os.makedirs(path)
#             print('Create experiment path successfully at %s' % path)