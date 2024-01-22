import pandas as pd
from datasets import Dataset
import torch

import json
from helpers import *

class MistralDataLoader():
    def __init__(self, tokenizer, prompt_function, max_length):
        self.tokenizer = tokenizer
        self.prompt_function = prompt_function
        self.max_length = max_length
        
        #self.train_stac = json.load(open('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/llama/experiments_finetune/data/train_stac.json'))
        #self.dev_stac = json.load(open('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/llama/experiments_finetune/data/dev_stac.json'))
        #self.test_stac = json.load(open('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/llama/experiments_finetune/data/test_stac.json'))

        self.id2label = {0: "(0) Comment", 1: "(1) Clarification question", 2: "(2) Question answer pair", 3: "(3) Continuation",
                    4: "(4) Acknowledgement", 5: "(5) Question elaboration", 6: "(6) Result", 7: "(7) Elaboration", 8: "(8) Explanation",
                    9: "(9) Correction", 10: "(10) Contrast", 11: "(11) Conditional", 12: "(12) Background", 13: "(13) Narration",
                    14: "(14) Alternation", 15: "(15) Parallel"}
        self.label2id = {v: k for k, v in self.id2label.items()}

    def process_dialog(self, dialog):
        # see Instruction Format here: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1

    def prepare_dataset(self, dataset):
        x,y = [],[]
        for elem in dataset:
            dialog = elem["dialog"]
            utterance_1 = f'[Turn {elem["sentence_one"]["speechturn"]}] {elem["sentence_one"]["speaker"]}: {elem["sentence_one"]["text"]}'
            utterance_2 = f'[Turn {elem["sentence_two"]["speechturn"]}] {elem["sentence_two"]["speaker"]}: {elem["sentence_two"]["text"]}'
            relation = elem["relation"]
            curr_dialog = self.prompt_function(dialog, utterance_1, utterance_2)

            processed_dialog = self.process_dialog(curr_dialog)

            x.append(processed_dialog)
            y.append(correct_relations[relation]) # TODO: map to gold-standard question instead

        return x,y

    def get_data(self):
        x_train,y_train = self.prepare_dataset(self.train_stac)
        x_dev,y_dev = self.prepare_dataset(self.dev_stac)
        x_test,y_test = self.prepare_dataset(self.test_stac)

        train_df = pd.DataFrame({'prompt': x_train, 'target': y_train})
        train_dataset = Dataset.from_pandas(train_df)

        dev_df = pd.DataFrame({'prompt': x_dev, 'target': y_dev})
        dev_dataset = Dataset.from_pandas(dev_df)

        test_df = pd.DataFrame({'prompt': x_test, 'target': y_test})
        test_dataset = Dataset.from_pandas(test_df)

        return train_dataset,dev_dataset,test_dataset
    
    def tokenize_dataset(self, dataset):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        combined_inputs = [f'{elem["prompt"]} {elem["target"]}' for elem in dataset]
        tokenized_inputs = self.tokenizer(combined_inputs, padding=True, truncation='only_first', max_length=self.max_length, return_tensors='pt')
        tokenized_inputs.labels = []

        tokenized_dataset = []
        for i,elem in enumerate(dataset):
            x,y = elem["prompt"],elem["target"]

            tokenized_prompt = self.tokenizer(x, return_tensors="pt")
            tokenized_target = self.tokenizer(y, return_tensors="pt")
            target_len,prompt_len = tokenized_target.input_ids.size(1),tokenized_prompt.input_ids.size(1)

            # set attention mask to 0 --> for text we want to predict
            tokenized_inputs.attention_mask[i][prompt_len:prompt_len+target_len] = 0
            # set label value to -100 --> for everything we want to ignore
            labels = torch.full(tokenized_inputs.input_ids[i].shape, -100)
            labels[prompt_len:prompt_len+target_len-1] = tokenized_inputs.input_ids[i][prompt_len:prompt_len+target_len-1]

            tokenized_dataset.append({
                "input_ids": tokenized_inputs.input_ids[i],
                "attention_mask": tokenized_inputs.attention_mask[i],
                "labels": labels
            })

        return tokenized_dataset