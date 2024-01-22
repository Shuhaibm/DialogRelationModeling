#!/bin/bash
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-15:0:0    
#SBATCH --mail-user=<mesm.shuhaib@gmail.com
#SBATCH --mail-type=ALL

cd ~/projects/def-vshwartz/shuhaibm

module load StdEnv/2020
module load gcc/9.3.0
module load arrow/8.0.0

source env_llama/bin/activate

cd DialogRelationModeling/experiments_finetune

for random_seed in 123 456 789; do
    python3 finetune_relations.py --model llama2 --random_seed $random_seed --prompt 3 --max_length 2048 --batch_size 1 --learning_rate 1e-4 --epochs 3
done