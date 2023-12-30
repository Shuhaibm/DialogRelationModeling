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


predictions = {}
for i,line in enumerate(curr_data_set.iter()):
    curr_len = len(predictions)
    for j,relation in enumerate(line["relations"]):
        key = str(curr_len + int(relation["y"])-1)
        key = "0"*(24-len(key)) + key
        val = str(int(relation["x"])+1)
        val = "XT" + "0"*(2-len(val)) + val

        predictions[key] = val

    
predictions_with_null = {}
for x in range(len(predictions)+1):
    key = str(x)
    key = "0"*(24-len(key)) + key

    if key in predictions:
        predictions_with_null[key] = predictions[key]
    else:
        predictions_with_null[key] = ""

f = open("predictions.json", "w")
f.write(json.dumps(predictions_with_null))