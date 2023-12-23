#!/bin/bash
#SBATCH -J env_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1 -w falcon6
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

module load any/python/3.8.3-conda
module load miniconda3
module load freeglut


conda activate vlnce

python --version
python env_test.py
