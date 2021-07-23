#!/bin/bash
#SBATCH --job-name="TrajTransformer"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/stud/rajr/storage/slurm/rajr/logs/rnn_baseline/slurm-%j.out
#SBATCH --error=/usr/stud/rajr/storage/slurm/rajr/logs/rnn_baseline/slurm-%j.out

source /usr/stud/rajr/anaconda3/bin/activate thesis
srun python src/position_maps/train_trajectory.py
