#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpuB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l22
export PATH=/vol/bitbucket/l22/finetuneenv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh

python src/models/hp_tuning.py

