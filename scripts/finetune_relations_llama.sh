#!/bin/bash
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=7-0:0:0   
#SBATCH --mail-user=<mesm.shuhaib@gmail.com
#SBATCH --mail-type=ALL

cd ~/projects/def-vshwartz/shuhaibm

module load StdEnv/2020
module load gcc/9.3.0
module load arrow/8.0.0

source env_llama/bin/activate

cd DialogRelationModeling/experiments_finetune


for seed in 123 789 456; do
    python3 finetune_relations.py --model_type llama2 --random_seed $seed --prompt 3 --max_length 2048 --batch_size 4 --learning_rate 5e-5 --epochs 5
done