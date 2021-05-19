#!/bin/bash
#SBATCH --job-name="TP_ObjClassifier"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=12G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/stud/rajr/storage/slurm/rajr/logs/rnn_baseline/slurm-%j.out
#SBATCH --error=/usr/stud/rajr/storage/slurm/rajr/logs/rnn_baseline/slurm-%j.out

source /usr/stud/rajr/anaconda3/bin/activate py38_tp
srun python baselinev2/improve_metrics/model.py
