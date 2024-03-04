#!/bin/bash
#SBATCH --mem=40G
#SBATCH --time=7-0:0:0   
#SBATCH --mail-user=<mesm.shuhaib@gmail.com
#SBATCH --mail-type=ALL

cd ~/projects/def-vshwartz/shuhaibm

module load StdEnv/2020
module load gcc/9.3.0
module load arrow/8.0.0

source env_llama/bin/activate

cd DialogRelationModeling/confusion_matrix


python3 chatgpt_confusion_matrix.py --prompt 3