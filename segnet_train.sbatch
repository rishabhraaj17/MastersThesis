#!/bin/bash
#SBATCH --job-name="UTP_SegNet_20ep"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=12G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/stud/rajr/storage/slurm/rajr/logs/vanilla_vgg_vae_1_epoch/slurm-%j.out
#SBATCH --error=/usr/stud/rajr/storage/slurm/rajr/logs/vanilla_vgg_vae_1_epoch/slurm-%j.out

source /usr/stud/rajr/anaconda3/bin/activate py38
srun python unsupervised_tp_0/deeplabv3.py --pin_memory --batch_norm --pretrained --num_workers 8 --batch_size 2 --epochs 80 --dataset_root /usr/stud/rajr/storage/user/TrajectoryPredictionMastersThesis/Datasets/SDD/ --save_path /usr/stud/rajr/storage/slurm/rajr/logs/vanilla_vgg_vae_1_epoch/checkpoint/
