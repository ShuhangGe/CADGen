#!/bin/bash

#SBATCH --nodes=1              # requests 1 compute server
#SBATCH --ntasks-per-node=1    # runs 1 task on each server 
#SBATCH --cpus-per-task=20      # uses 2 compute cores per task
#SBATCH --time=24:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=auto
#SBATCH --output=auto.out
#SBATCH --gres=gpu:rtx8000:1

module purge

singularity exec --nv \
                    --overlay /scratch/sg7484/pytorch-example/CMDGen-15GB.ext3:ro \
                            /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
                             /bin/bash -c "source /ext3/env.sh; CUDA_LAUNCH_BLOCKING=1 python main_parameter_autogressive.py \
                             --save_path /scratch/sg7484/CADGen/bulletpoints/mae_cad/output_autogressive/1e-4_075_all \
                             --lr 1e-4 --mask_ratio 0.75"