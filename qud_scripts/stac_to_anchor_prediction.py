import jsonlines
import os
import shutil
import json
from collections import OrderedDict


stac_dev = jsonlines.open('data/stac/dev_subindex.json')
stac_test = jsonlines.open('data/stac/test_subindex.json')
stac_train = jsonlines.open('data/stac/train_subindex.json')

# Select the dataset
curr_data_set = stac_test

inputa_path = "./data/inputa/"
if os.path.exists(inputa_path):
    shutil.rmtree(inputa_path)
path = os.mkdir(inputa_path)

for i,line in enumerate(curr_data_set.iter()):
    f = open(inputa_path+str(i), "w")
    for j,edu in enumerate(line["edus"]):
        edu_context = f'{edu["speaker"]}: {edu["text"]}'
        
        if j == len(line["edus"]) - 1:
            f.write(f'{str(j+1)}\t{edu_context}')
        else:
            f.write(f'{str(j+1)}\t{edu_context}\n')