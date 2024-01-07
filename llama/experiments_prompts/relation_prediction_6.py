# Do not input whole dialog. Define every relation

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog

import json
import jsonlines

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


    stac_test = jsonlines.open('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/data/stac/test_subindex.json')
    stac_dialogs = {}
    for i,line in enumerate(stac_test.iter()):
        curr_id = line["id"]
        context = ""
        for j,edu in enumerate(line["edus"]):
            edu_message = f'[Turn {j}] {edu["speaker"]}: {edu["text"]}'
            context += edu_message + "\n"
        
        stac_dialogs[curr_id] = context

    system_message = {
        "role": "system", 
        "content": "The elaboration relation describes when the second utterance provides more information about the eventuality introduced in the first utterance. The explanation relation describes when the second utterance explains the cause of what happened in the first utterance. The acknowledgement relation describes when the second utterance signals an understanding or acceptance of the first utterance. The question answer pair relation describes when the second utterance gives an answer to a question in the first utterance. The question elaboration relation describes when the second utterance is a follow-up question intended to get more information to answer the question presented in the first utterance. The clarification question relation describes when the second utterance tries to clarify what was said in the first utterance. The comment relation describes when the second utterance provides an opinion or evaluation of the first utterance. The narration relation describes when main eventualities of the first and second utterances occur in sequence. The continuation relation describes when the first and second utterances elaborate or provide background to the same segment. The contrast relation describes when two utterances have similar semantic structures, but contrasting themes. The parallel relation describes when two utterances have similar semantic structures and similar themes. The result relation describes when the main eventuality of the first utterance is understood to cause the eventuality given in the second utterance. The background relation describes when the second utterance provides some stage setting for the event that happened in the first utterance. The conditional relation describes when the first utterance is a hypothesis and the second utterance is a consequence of the hypothesis. The alternation relation describes when there is a disjunction between two utterances. The correction relation describes when the second utterance corrects the first utterance. Only respond with one of the following relations: elaboration, explanation, acknowledgement, question answer pair, question elaboration, clarification question, comment, narration, continuation, contrast, parallel, result, background, conditional, alternation, correction."
    }

    correct_relations = {"Elaboration": ["elaboration"],
                          "Explanation": ["explanation"],
                          "Acknowledgement": ["acknowledgement"],
                          "Question_answer_pair": ["question answer pair"],
                          "Q_Elab": ["question elaboration"],
                          "Clarification_question": ["clarification question"],
                          "Comment": ["comment"],
                          "Narration": ["narration"],
                          "Continuation": ["continuation"],
                          "Contrast": ["contrast"],
                          "Parallel": ["parallel"],
                          "Result": ["result"],
                          "Background": ["background"],
                          "Conditional": ["conditional"],
                          "Alternation": ["alternation"],
                          "Correction": ["correction"]
                          }
    relation_to_question = {"Elaboration": "How is the initial statement or situation further detailed or expanded upon?",
                          "Explanation": "What is the reason or cause for this situation?",
                          "Acknowledgement": "Is there an acknowledgment in response to the previous statement or situation?",
                          "Question_answer_pair": "What is the response to the previously asked question?",
                          "Q_Elab": "What further clarification or information is being sought with the follow-up question?",
                          "Clarification_question": "Can you provide more details or clarify what you just mentioned?",
                          "Comment": "How is the content or situation in the previous statement being personally evaluated or commented on?",
                          "Narration": "What happens next in the sequence of events?",
                          "Continuation": "What additional information or details are provided on the same topic?",
                          "Contrast": "Is there a statement that contrasts or differs in theme or consequence from the first statement?",
                          "Parallel": "Is there a statement that shares a common theme or idea with the first statement?",
                          "Result": "What is the effect or result caused by the first statement?",
                          "Background": "What background or context information sets the stage for the main event or situation?",
                          "Conditional": "What will be the outcome if a certain condition is met?",
                          "Alternation": "What are the different options or alternatives presented?",
                          "Correction": "Is the first statement being corrected or refuted as factual or accurate?"}

    test_stac_qud = json.load(open('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/qud_question_classification/data/test_stac_qud.json'))

    total,correct,exceptions = 0,0,0
    for elem in test_stac_qud:
        total += 1

        curr_dialog = [system_message]

        user_message = "In a dialogue, generate a question that represents the relation between " + f'[ Turn {elem["sentence_one"]["speechturn"]}] {elem["sentence_one"]["speaker"]}: {elem["sentence_one"]["text"]} and' + f'[ Turn {elem["sentence_two"]["speechturn"]}] {elem["sentence_two"]["speaker"]}: {elem["sentence_two"]["text"]}'
        curr_dialog.append({"role": "user", "content":user_message})
        assistant_message = relation_to_question[elem["relation"]]
        curr_dialog.append({"role": "assistant", "content":assistant_message})
        user_message = "Now predict the relation between the two utterances."
        curr_dialog.append({"role": "user", "content":user_message})
        

        
        try:
            results = generator.chat_completion(
                [curr_dialog],  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            i = 0
            for result in results:
                print(f'The correct relation is {elem["relation"]}')
                i += 1
                for msg in curr_dialog:
                    print(f"{msg['role'].capitalize()}: {msg['content']}\n")
                print(
                    f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                )
                print("\n==================================\n")

                for correct_relation in correct_relations[elem["relation"]]:
                    if correct_relation.lower() in result['generation']['content'].lower():
                        correct += 1
        except:
            exceptions += 1
            print("An exception occurred")
    
    print(f'total: {total}, correct: {correct}, exceptions: {exceptions}')

if __name__ == "__main__":
    fire.Fire(main)
