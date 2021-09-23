#!/bin/bash
#SBATCH --job-name="ExtractLocations"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/stud/rajr/storage/slurm/rajr/logs/rnn_baseline/slurm-%j.out
#SBATCH --error=/usr/stud/rajr/storage/slurm/rajr/logs/rnn_baseline/slurm-%j.out

source /usr/stud/rajr/anaconda3/bin/activate thesis
srun python src/position_maps/locations_collector.py
