#!/bin/bash
#
#SBATCH --partition=disc
#SBATCH --nodelist=c851
#SBATCH --mem=15G
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=2
#SBATCH --output=out_shadow_former.txt
#SBATCH --error=error_shadow_former.txt
#SBATCH --time=05:10:00

nvidia-smi
python train.py  --win_size 8 --train_ps 256 --batch_size 16 --train_dir /home/ahsan/shadow_trial/ISTD_Dataset/train --val_dir /home/ahsan/shadow_trial/ISTD_Dataset/test

