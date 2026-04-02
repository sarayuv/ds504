#!/bin/bash

#SBATCH -A cs586
#SBATCH -p academic
#SBATCH --job-name=gan_project3
#SBATCH --output=project3_output.log
#SBATCH --error=project3_error.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

# Load CUDA so PyTorch can use the GPU.
module load cuda

# Activate your conda environment.
source ~/.bashrc
conda activate project3_env

# Move into the project3 directory.
cd ~/ds504/project3

# Run GAN training.
python train.py
