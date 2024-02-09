import pandas as pd
from datasets import Dataset
import torch

import json
from models.FollowupDataLoader import FollowupDataLoader

class LlamaFollowupDataLoader(FollowupDataLoader):
    def __init__(self, tokenizer, prompt_function, max_length, dataset):
        super().__init__(tokenizer, prompt_function, max_length, dataset)

    def process_dialog(self, dialog):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        BOS, EOS = "<s>","</s>"

        if dialog[0]["role"] == "system":
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
        
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )

        dialog_final = ""
        for prompt, answer in zip(dialog[::2],dialog[1::2]):
            curr_elem = f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
            dialog_final += curr_elem
        
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"

        dialog_final += f"{BOS}{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"

        return dialog_final
