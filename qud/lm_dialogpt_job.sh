#!/bin/bash
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=15:0:0    
#SBATCH --mail-user=<mesm.shuhaib@gmail.com
#SBATCH --mail-type=ALL

cd ~/projects/def-vshwartz/shuhaibm/DialogRelationModeling/qud_question_classification/classifiers
source ../../../env/bin/activate

for run in "1" "2" "3" "4" "5" "6" "7" "8"; do
    python3 dialogpt_classifier.py "2e-5" $run
done
