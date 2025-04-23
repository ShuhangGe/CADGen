#!/bin/bash

#SBATCH --nodes=1              # requests 1 compute server
#SBATCH --ntasks-per-node=1    # runs 1 task on each server 
#SBATCH --cpus-per-task=20      # uses 2 compute cores per task
#SBATCH --time=24:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=para1
#SBATCH --output=para1.out
#SBATCH --gres=gpu:rtx8000:1

module purge

singularity exec --nv \
                    --overlay /scratch/sg7484/pytorch-example/python39.ext3:ro \
                            /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
                             /bin/bash -c "source /ext3/env.sh; python show.py \
                             --src /scratch/sg7484/CADGen/bulletpoints/mae_cad/decoder_result --with_gt \
                             --idx 0 --num 10"