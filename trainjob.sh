#!/bin/bash
#SBATCH -J VLN_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:2
#SBATCH -t 7-23:59 
#SBATCH --cpus-per-task=24
#SBATCH --mem=50GB

module load any/python/3.8.3-conda
module load miniconda3
module load freeglut


conda activate vlnce

python --version
python run.py  --exp-config vlnce_baselines/config/paper_configs/intuition_config/cma_pm_aug.yaml --run-type train