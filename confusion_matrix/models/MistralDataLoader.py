import pandas as pd
from datasets import Dataset
import torch

import json
from DataLoader import DataLoader

class MistalDataLoader(DataLoader):
    def __init__(self, tokenizer, prompt_function, max_length):
        super().__init__(self, tokenizer, prompt_function, max_length)
    def process_dialog(self, dialog):
        # see Instruction Format here: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
        return dialog
        pass