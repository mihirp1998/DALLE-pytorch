#!/usr/bin/env bash
#SBATCH --partition=kate_reserved
#SBATCH --job-name=digen
#SBATCH --output=logs/%A.out
#SBATCH --error=logs/%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --exclude=matrix-0-18,matrix-0-16,matrix-0-22,matrix-0-36
#SBATCH --nodelist=matrix-3-18
source /home/mprabhud/.bashrc
conda activate vlr
cd /home/mprabhud/sp/DALLE-pytorch
export PYTHONUNBUFFERED=1

# python run.py data=imagenet model=mlp batch_size=80 learning_rate=2e-2 mode=forward_forward
# python run.py data=coco model=mlp num_encoder_layers=48 emb_size=768 mode=forward_forward
python train_dalle.py exp=ff train_test_split=0.8