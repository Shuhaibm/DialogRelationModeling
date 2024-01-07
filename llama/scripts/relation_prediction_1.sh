#!/bin/bash
#SBATCH --mem=40G
#SBATCH --gres=gpu:v100l:1
#SBATCH --time=15:0:0    
#SBATCH --mail-user=<mesm.shuhaib@gmail.com
#SBATCH --mail-type=ALL

cd ~/projects/def-vshwartz/shuhaibm

module load StdEnv/2020
module load gcc/9.3.0
module load arrow/8.0.0
module load python/3.10

source env_llama/bin/activate

cd DialogRelationModeling/llama/experiments_finetune

python3 relation_prediction_finetune_1.py