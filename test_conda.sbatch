#!/bin/bash
#
#SBATCH --partition=normal         # Partition with GPU resources
#SBATCH --nodelist=c735           # Specify the node
#SBATCH --gres=gpu:A100:1         # Request one A100 GPU           # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G   # Memory allocation
#SBATCH --output=out_shadow_former.txt
#SBATCH --error=error_shadow_former.txt
#SBATCH --time=05:10:00

nvidia-smi
python train.py  --win_size 8 --train_ps 256 --batch_size 16 --train_dir /home/ahsan/shadow_trial/ISTD_Dataset/train --val_dir /home/ahsan/shadow_trial/ISTD_Dataset/test

