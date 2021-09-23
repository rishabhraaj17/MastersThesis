#!/bin/bash
#SBATCH --job-name="TracksAnalysis"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0,VRAM:16G
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/stud/rajr/storage/slurm/rajr/logs/rnn_baseline/slurm-%j.out
#SBATCH --error=/usr/stud/rajr/storage/slurm/rajr/logs/rnn_baseline/slurm-%j.out

source /usr/stud/rajr/anaconda3/bin/activate thesis
srun python baselinev2/notebooks/dataset_analysis.py
