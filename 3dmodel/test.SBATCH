#!/bin/bash

#SBATCH --nodes=1              # requests 1 compute server
#SBATCH --ntasks-per-node=1    # runs 1 task on each server 
#SBATCH --cpus-per-task=20      # uses 2 compute cores per task
#SBATCH --time=96:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=mae0
#SBATCH --output=mae0.out
#SBATCH --gres=gpu:rtx8000:1

module purge

singularity exec --nv \
                    --overlay /scratch/sg7484/pytorch-example/CMDGen-15GB-500K.ext3:ro \
                            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
                             /bin/bash -c "source /ext3/env.sh; python test.py --epochs 900 --lr 1e-5\
                             --train_batch 512 --test_batch 128 --num_decoder_layers_deformable 3 --unet_dim 4 --resnet 18\
                                --data_root /scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1 --cad_root /scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/cad \
                             --cmd_root /scratch/sg7484/data/CMDGen/Sketch_1_Extrude_1/cmd --pic_root /scratch/sg7484/data/CMDGen/data3D_pic_combine"
