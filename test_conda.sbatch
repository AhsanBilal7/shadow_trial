#!/bin/bash
#
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1         # Request two  A100 GPUs           # Request 1 GPU
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G   # Memory allocation
#SBATCH --output=outaug24_shadow_former.txt
#SBATCH --error=erraug24_shadow_former.txt
#SBATCH --time=10:10:00

nvidia-smi
python train.py  --win_size 8 --train_ps 256 --batch_size 32 --train_dir /home/ahsan/shadow_trial/ISTD_Dataset/train --val_dir /home/ahsan/shadow_trial/ISTD_Dataset/test

