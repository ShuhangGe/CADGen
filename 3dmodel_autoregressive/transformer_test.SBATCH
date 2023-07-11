#!/bin/bash

#SBATCH --nodes=1              # requests 1 compute server
#SBATCH --ntasks-per-node=1    # runs 1 task on each server 
#SBATCH --cpus-per-task=20      # uses 2 compute cores per task
#SBATCH --time=10:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=test
#SBATCH --output=/scratch/sg7484/CMDGen/results/out/test.out
#SBATCH --gres=gpu:rtx8000:1

module purge

singularity exec --nv \
                    --overlay /scratch/sg7484/pytorch-example/CMDGen-15GB-500K.ext3:ro \
                            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
                             /bin/bash -c "source /ext3/env.sh; python train_transform.py  --train_batch 10 --lr 1e-5 --save_path ../results/test --epochs 1000 "
