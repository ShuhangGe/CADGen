#!/bin/bash

#SBATCH --nodes=1              # requests 1 compute server
#SBATCH --ntasks-per-node=1    # runs 1 task on each server 
#SBATCH --cpus-per-task=20      # uses 2 compute cores per task
#SBATCH --time=24:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=trans3
#SBATCH --output=/scratch/sg7484/CMDGen/results/out/trans3.out
#SBATCH --gres=gpu:rtx8000:1

module purge

singularity exec --nv \
                    --overlay /scratch/sg7484/pytorch-example/CMDGen-15GB-500K.ext3:ro \
                            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
                             /bin/bash -c "source /ext3/env.sh; python train_transform.py  --train_batch 100 --lr 1e-7 --num_decoder_layers_transdecoder 4 --save_path ../results/auto/fulldata_auto_transformer_1e-7_res18_trans4 --epochs 1000 --data_root /scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1 \
                             --cad_root /scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/cad --cmd_root /scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/cmd --pic_root /scratch/sg7484/data/CMDGen/data3D_pic_combine"
