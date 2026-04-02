#!/bin/bash

#SBATCH -A cs586
#SBATCH -p academic
#SBATCH --job-name=gan_project3
#SBATCH --output=project3_output_%j.log
#SBATCH --error=project3_error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

# Load a CUDA module compatible with torch 2.5.1+cu121.
module load cuda12.1/toolkit/12.1.1

# Initialize Conda and activate the environment.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate project3_env

# Move to the parent directory and run training.
cd ~/ds504
python project3/train.py
