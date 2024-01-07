import jsonlines
import os
import shutil
import json
from collections import OrderedDict
from relation_prediction_finetune_helpers import *

stac_dialogs = get_stac_dialogs()
test_stac_qud = json.load(open('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/qud/qud_question_classification/data/test_stac_qud.json'))
train_stac_qud = json.load(open('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/qud/qud_question_classification/data/train_stac_qud.json'))
dev_stac_qud = json.load(open('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/qud/qud_question_classification/data/dev_stac_qud.json'))

test_stac = []
for elem in test_stac_qud:
    del elem['question']
    elem["dialog"] = stac_dialogs[elem["id"]]
    test_stac.append(elem)

f = open("./data/test_stac.json", "w")
f.write(json.dumps(test_stac))

dev_stac = []
for elem in dev_stac_qud:
    del elem['question']
    elem["dialog"] = stac_dialogs[elem["id"]]
    dev_stac.append(elem)

f = open("./data/dev_stac.json", "w")
f.write(json.dumps(dev_stac))

train_stac = []
for elem in train_stac_qud:
    del elem['question']
    elem["dialog"] = stac_dialogs[elem["id"]]
    train_stac.append(elem)

f = open("./data/train_stac.json", "w")
f.write(json.dumps(train_stac))