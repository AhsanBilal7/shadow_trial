#!/bin/bash
#
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1         # Request two  A100 GPUs           # Request 1 GPU
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G   # Memory allocation
#SBATCH --output=v1_out.txt
#SBATCH --error=v1_error.txt
#SBATCH --time=40:10:00

nvidia-smi
python train.py --warmup --win_size 10 --train_ps 320 --batch_size 16 --train_dir /home/ahsan/shadow_trial/ISTD_Dataset/train --val_dir /home/ahsan/shadow_trial/ISTD_Dataset/test --save_dir /home/ahsan/shadow_trial/log


