# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog

from relation_prediction_helpers import *
import json

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    stac_dialogs = get_stac_dialogs()

    test_stac_qud = json.load(open('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/qud_question_classification/data/test_stac_qud.json'))

    total,correct,exceptions = 0,0,0
    for elem in test_stac_qud:
        dialog = stac_dialogs[elem["id"]]
        utterance_1 = f'[Turn {elem["sentence_one"]["speechturn"]}] {elem["sentence_one"]["speaker"]}: {elem["sentence_one"]["text"]}'
        utterance_2 = f'[Turn {elem["sentence_two"]["speechturn"]}] {elem["sentence_two"]["speaker"]}: {elem["sentence_two"]["text"]}'
        qud_question = elem["question"]
        relation = elem["relation"]
        curr_dialog = get_prompt_8(dialog, utterance_1, utterance_2)
        

        result = generator.chat_completion(
            [curr_dialog],  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )[0]

        if True: # follow-up question
            
            assistant_message = {"role": result['generation']['role'], "content": result['generation']['content']}
            curr_dialog.append(assistant_message)
            curr_dialog.append(get_follow_up_prompt_8(utterance_1, utterance_2))

            result = generator.chat_completion(
                [curr_dialog],  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )[0]
        
        if True: # follow-up question 2
            assistant_message = {"role": result['generation']['role'], "content": result['generation']['content']}
            curr_dialog.append(assistant_message)
            curr_dialog.append({"role": "user", "content": f'Are you sure? Refer to the list of relations and their definitions. Once you have your final answer, respond with one of the labels, without adding any additional information or context.'})

            result = generator.chat_completion(
                [curr_dialog],  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )[0]
        
        total += 1
        for msg in curr_dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )


        if correct_relations[elem["relation"]].strip() in result['generation']['content'].strip():
            correct += 1
            print("\nThe answer is correct")
        else:
            print(f'\nThe answer is incorrect, you predicted {result["generation"]["content"]}, but the correct answer is {correct_relations[elem["relation"]]}')

        print("\n==================================\n")

    print(f'total: {total}, correct: {correct}, exceptions: {exceptions}')

if __name__ == "__main__":
    fire.Fire(main)
