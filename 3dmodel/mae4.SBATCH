#!/bin/bash

#SBATCH --nodes=1              # requests 1 compute server
#SBATCH --ntasks-per-node=1    # runs 1 task on each server 
#SBATCH --cpus-per-task=20      # uses 2 compute cores per task
#SBATCH --time=24:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=mae0
#SBATCH --output=/scratch/sg7484/CMDGen/results/out/train0_fulldata_noauto_deformable4_res50_unet3_1e-4.out
#SBATCH --gres=gpu:rtx8000:1

module purge

singularity exec --nv \
                    --overlay /scratch/sg7484/pytorch-example/CMDGen-15GB-500K.ext3:ro \
                            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
                             /bin/bash -c "source /ext3/env.sh; python train_deformable.py --lr 1e-4 --train_batch 100 --num_decoder_layers_deformable 4 --save_path ../results/noauto/deformable4_lrtest/fulldata_noauto_deformable4_res50_unet3_1e-4 --epochs 900 --data_root /scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1 \
                             --cad_root /scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/cad --cmd_root /scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/cmd --pic_root /scratch/sg7484/data/CMDGen/data3D_pic_combine"
