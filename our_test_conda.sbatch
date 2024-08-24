#!/bin/bash
#
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1         # Request two  A100 GPUs           # Request 1 GPU
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G   # Memory allocation
#SBATCH --output=our_outaug24_shadow_former.txt
#SBATCH --error=our_erraug24_shadow_former.txt
#SBATCH --time=10:10:00

nvidia-smi
python train.py  --win_size 8 --train_ps 320 --batch_size 128 --train_dir /home/ahsan/shadow_trial/ISTD_Dataset/train --val_dir /home/ahsan/shadow_trial/ISTD_Dataset/$

