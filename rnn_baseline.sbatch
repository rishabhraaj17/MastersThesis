#!/bin/bash
#SBATCH --job-name="RecurrentBaseline"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1,VRAM:4G
#SBATCH --mem=12G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/stud/rajr/storage/slurm/rajr/logs/rnn_baseline/slurm-%j.out
#SBATCH --error=/usr/stud/rajr/storage/slurm/rajr/logs/rnn_baseline/slurm-%j.out

source /usr/stud/rajr/anaconda3/bin/activate py38
srun python baselinev2/nn/train.py
