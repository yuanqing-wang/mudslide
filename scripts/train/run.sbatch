#!/bin/bash
#SBATCH --time=47:59:00
#SBATCH --mem=20GB
#SBATCH --job-name=qm9
#SBATCH --output=%A.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 run.py
python run.py
