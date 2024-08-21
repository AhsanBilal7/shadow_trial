#!/bin/bash
#
#SBATCH --partition=gpu           # Partition with GPU resources
#SBATCH --gres=gpu:2              # Request 1 GPU
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --cpus-per-task=10         # Number of CPU cores per task
#SBATCH --mem=15G                  # Memory allocation
#SBATCH --output=out_shadow_former.txt
#SBATCH --error=error_shadow_former.txt
#SBATCH --time=24:10:00
nvidia-smi
python train.py  --win_size 8 --train_ps 256 --batch_size 16 --train_dir /home/ahsan/shadow_trial/ISTD_Dataset/train --val_dir /home/ahsan/shadow_trial/ISTD_Dataset/test

