#!/bin/bash

#SBATCH --nodes=1              # requests 1 compute server
#SBATCH --ntasks-per-node=1    # runs 1 task on each server 
#SBATCH --cpus-per-task=20      # uses 2 compute cores per task
#SBATCH --time=96:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=mae2
#SBATCH --output=mae2.out
#SBATCH --gres=gpu:rtx8000:1

module purge

singularity exec --nv \
                    --overlay /scratch/sg7484/pytorch-example/CMDGen-15GB-500K2.ext3:rw \
                            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
                             /bin/bash -c "source /ext3/env.sh; python train.py --lr 1e-5 --save_path ../results/train_1e-5_fulldata_autoregressive_maskedpadding --epochs 900 --data_root /scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1 \
                             --cad_root /scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/cad --cmd_root /scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/cmd --pic_root /scratch/sg7484/data/CMDGen/data3D_pic_combine"
