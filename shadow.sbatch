#!/bin/bash
#
#SBATCH --partition=debug_gpu
#SBATCH --nodelist=c299
#SBATCH --gres=gpu:kepler:2         # Request one A100 GPU           # Request 1 GPU
#SBATCH --mem=15G
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=2
#SBATCH --output=out_shadow_former.txt
#SBATCH --error=error_shadow_former.txt
#SBATCH --time=05:10:00

nvidia-smi
python train.py  --win_size 8 --train_ps 256 --batch_size 16 --train_dir /home/ahsan/shadow_trial/ISTD_Dataset/train --val_dir /home/ahsan/shadow_trial/ISTD_Dataset/test

