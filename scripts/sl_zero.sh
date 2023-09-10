#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpuB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l22
export PATH=/vol/bitbucket/l22/finetuneenv/bin/:$PATH
source activate

python src/models/run_benchmarks_zero_shot.py