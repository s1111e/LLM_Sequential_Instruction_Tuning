#!/bin/bash
#SBATCH --job-name=stage2-lr1e5_llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --output=stage2-lr1e5.out
#SBATCH --error=stage2-lr1e5.err
#SBATCH --partition=gpu1v100
#SBATCH --gres=gpu:1

echo "===== START ====="
date
hostname
nvidia-smi

# Conda load
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate /work/amd456/llm_clean_env

echo "===== ENV READY ====="
which python
python --version

# Cache 
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/work/amd456/.cache/huggingface

cd /work/amd456/LLM/HW3


echo "===== STAGE 2 TRAIN ====="

accelerate launch \
  --config_file accelerate-config-v2-ds.yaml \
  train-sft.py

echo "===== DONE ====="