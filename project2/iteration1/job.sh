#!/bin/bash

#SBATCH -A cs586
#SBATCH -p academic
#SBATCH --job-name=taxi_iter1
#SBATCH --output=iter1_output.log
#SBATCH --error=iter1_error.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Conda is already available on Turing — no module load needed for python
# Load CUDA so PyTorch can find the GPU
module load cuda

# Activate your conda environment (replace "myenv" with your env name)
source ~/.bashrc
conda activate myenv

# Move into your project directory (replace with your actual path)
cd ~/project2/iteration1

# Run training via the required entry point
python main.py train