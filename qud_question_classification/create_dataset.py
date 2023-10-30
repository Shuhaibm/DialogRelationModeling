import jsonlines
import os
import shutil
import json
from collections import OrderedDict


stac_dev = jsonlines.open('../data/stac/dev_subindex.json')
stac_test = jsonlines.open('../data/stac/test_subindex.json')
stac_train = jsonlines.open('../data/stac/train_subindex.json')

# dev_stac_qud = json.load(open('./data/dev_stac_qud.json'))
test_stac_qud = json.load(open('./data/test_stac_qud.json'))
train_stac_qud = json.load(open('./data/train_stac_qud.json'))

# Select the dataset
curr_data_set = stac_dev

# count = 0
# for i,line in enumerate(curr_data_set.iter()):
#     for relation in line["relations"]:
#         test_stac_qud[count]["relation"] = relation["type"]
#         count += 1

# f = open("test_stac_qud.json", "w")
# f.write(json.dumps(test_stac_qud))



data = []

for i,line in enumerate(curr_data_set.iter()):
    for relation in line["relations"]:
        curr_relation = {}

        x,y = min(relation["x"],relation["y"]), max(relation["x"],relation["y"])
        curr_relation["sentence_one"] = line["edus"][x]
        curr_relation["sentence_two"] = line["edus"][y]
        curr_relation["relation"] = relation["type"]
        
        data.append(curr_relation)
    
with open("results.txt") as f: # TODO: change results as needed
    for i,line in enumerate(f):
        data[i]["question"] = line


f = open("dev_stac_qud.json", "w")
f.write(json.dumps(data))