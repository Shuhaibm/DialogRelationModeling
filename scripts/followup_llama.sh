#!/bin/bash
#SBATCH --mem=40G
#SBATCH --gres=gpu:v100l:1
#SBATCH --time=5-15:0:0   
#SBATCH --mail-user=<mesm.shuhaib@gmail.com
#SBATCH --mail-type=ALL

cd ~/projects/def-vshwartz/shuhaibm

module load StdEnv/2020
module load gcc/9.3.0
module load arrow/8.0.0

source env_llama/bin/activate

cd DialogRelationModeling/experiments_finetune


python3 followup.py --model_type llama2 --random_seed 123 --prompt 8 --max_length 2048 --size 5000