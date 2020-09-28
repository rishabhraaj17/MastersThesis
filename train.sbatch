#!/bin/bash
#SBATCH --job-name="UTP_VanillaAE_VGG"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=12G
#SBATCH --time=05:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/stud/rajr/storage/slurm/rajr/logs/vanilla_vgg_vae_1_epoch/slurm-%j.out
#SBATCH --error=/usr/stud/rajr/storage/slurm/rajr/logs/vanilla_vgg_vae_1_epoch/slurm-%j.out

source /usr/stud/rajr/anaconda3/bin/activate py38
srun python unsupervised_tp_0/train.py
