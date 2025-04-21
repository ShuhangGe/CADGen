#!/bin/bash

#SBATCH --nodes=1              # requests 1 compute server
#SBATCH --ntasks-per-node=1    # runs 1 task on each server 
#SBATCH --cpus-per-task=20      # uses 2 compute cores per task
#SBATCH --time=48:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=class
#SBATCH --output=class.out
#SBATCH --gres=gpu:rtx8000:1

module purge

singularity exec --nv \
                    --overlay /scratch/sg7484/pytorch-example/CMDGen-15GB.ext3:ro \
                            /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
                             /bin/bash -c "source /ext3/env.sh; CUDA_LAUNCH_BLOCKING=1 python main_parameter_mask_class.py \
                             --lr 1e-4 --save_path /scratch/sg7484/CADGen/bulletpoints/mae_cad/output_class/test \
                             --mask_ratio 0.5 --decoder_depth 8 --decoder_embed_dim 512 --depth 12 --embed_dim 512"