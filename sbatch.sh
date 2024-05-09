#!/usr/bin/env bash
#SBATCH --partition=kate_reserved
#SBATCH --job-name=d_i_c
#SBATCH --output=/home/mprabhud/sp/DALLE-pytorch/logs/%A.out
#SBATCH --error=/home/mprabhud/sp/DALLE-pytorch/logs/%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --exclude=matrix-0-18,matrix-0-16,matrix-0-22,matrix-0-36,matrix-1-22
#SBATCH --nodelist=matrix-1-24
source /home/mprabhud/.bashrc
conda activate vlr_sp2
cd /home/mprabhud/sp/DALLE-pytorch/
export PYTHONUNBUFFERED=1

# python run.py data=imagenet model=mlp batch_size=80 learning_rate=2e-2 mode=forward_forward
# python run.py data=coco model=mlp num_encoder_layers=48 emb_size=768 mode=forward_forward
# python train_dalle.py +hydra/launcher=matrix exp=ff,r learning_rate=2e-3,3e-4
python train_dalle.py exp=ff learning_rate=3e-4