#!/bin/bash
#
#SBATCH --partition=debug           # Partition with GPU resources
#SBATCH --gres=gpu:2              # Request 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G   # Memory allocation
#SBATCH --output=out_shadow_former.txt
#SBATCH --error=error_shadow_former.txt
#SBATCH --time=12:10:00

nvidia-smi
python train.py  --win_size 8 --train_ps 256 --batch_size 16 --train_dir /home/ahsan/shadow_trial/ISTD_Dataset/train --val_dir /home/ahsan/shadow_trial/ISTD_Dataset/test

