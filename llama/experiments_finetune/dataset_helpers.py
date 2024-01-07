from relation_prediction_finetune_helpers import *

def prepare_dataset(dataset, prompt_function):
    x,y = [],[]
    for elem in dataset:
        dialog = elem["dialog"]
        utterance_1 = f'[Turn {elem["sentence_one"]["speechturn"]}] {elem["sentence_one"]["speaker"]}: {elem["sentence_one"]["text"]}'
        utterance_2 = f'[Turn {elem["sentence_two"]["speechturn"]}] {elem["sentence_two"]["speaker"]}: {elem["sentence_two"]["text"]}'
        relation = elem["relation"]
        curr_dialog = prompt_function(dialog, utterance_1, utterance_2)

        processed_dialog = process_dialog(curr_dialog)

        x.append(processed_dialog)
        y.append(correct_relations[relation])
    
    id2label = {0: "(0) Comment", 1: "(1) Clarification question", 2: "(2) Question answer pair", 3: "(3) Continuation",
                4: "(4) Acknowledgement", 5: "(5) Question elaboration", 6: "(6) Result", 7: "(7) Elaboration", 8: "(8) Explanation",
                9: "(9) Correction", 10: "(10) Contrast", 11: "(11) Conditional", 12: "(12) Background", 13: "(13) Narration",
                14: "(14) Alternation", 15: "(15) Parallel"}
    label2id = {v: k for k, v in id2label.items()}
    
    return x,y,id2label,label2id


def process_dialog(dialog):
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
        curr_elem = f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}",
        dialog_final += curr_elem
    
    assert (
        dialog[-1]["role"] == "user"
    ), f"Last message must be from user, got {dialog[-1]['role']}"

    dialog_final += f"{BOS}{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"


    return dialog_final


def get_prompt_3(dialog, utterance_1, utterance_2):
    system_message = {
        "role": "system", 
        "content": f"""Respond with only one of the following labels. The labels are listed below, along with their definitions:
                (0) Comment: when the second utterance provides an opinion or evaluation of the first utterance
                (1) Clarification question: when the second utterance tries to clarify what was said in the first utterance
                (2) Question answer pair: when the second utterance gives an answer to a question in the first utterance
                (3) Continuation: when the first and second utterances elaborate or provide background to the same segment
                (4) Acknowledgement: when the second utterance signals an understanding or acceptance of the first utterance
                (5) Question elaboration: when the second utterance is a follow-up question intended to get more information to answer the question presented in the first utterance
                (6) Result: when the main eventuality of the first utterance is understood to cause the eventuality given in the second utterance
                (7) Elaboration: when the second utterance provides more information about the eventuality introduced in the first utterance
                (8) Explanation: when the second utterance explains the cause of what happened in the first utterance
                (9) Correction: when the second utterance corrects the first utterance
                (10) Contrast: when two utterances have similar semantic structures, but contrasting themes
                (11) Conditional: when the first utterance is a hypothesis and the second utterance is a consequence of the hypothesis
                (12) Background: when the second utterance provides some stage setting for the event that happened in the first utterance
                (13) Narration: when main eventualities of the first and second utterances occur in sequence
                (14) Alternation: when there is a disjunction between two utterances
                (15) Parallel: when two utterances have similar semantic structures and similar themes
Please respond with one of the labels, without adding any additional information or context."""
    }

    user_message = {
        "role": "user",
        "content": f'Given the following dialog: \n{dialog} \nWhat is the relation between \'{utterance_1}\' and \'{utterance_2}\'? Please provide only the label that best fits your response, and refrain from including any extra details or examples.'
    }

    return [system_message, user_message]

def get_prompt_8(dialog, utterance_1, utterance_2): # labels, definitions, dialog, utterances
    system_message = {
        "role": "system", 
        "content": f"""The labels for every relation are listed below, along with their definitions:
                (0) Comment: when the second utterance provides an opinion or evaluation of the first utterance
                (1) Clarification question: when the second utterance tries to clarify what was said in the first utterance
                (2) Question answer pair: when the second utterance gives an answer to a question in the first utterance
                (3) Continuation: when the first and second utterances elaborate or provide background to the same segment
                (4) Acknowledgement: when the second utterance signals an understanding or acceptance of the first utterance
                (5) Question elaboration: when the second utterance is a follow-up question intended to get more information to answer the question presented in the first utterance
                (6) Result: when the main eventuality of the first utterance is understood to cause the eventuality given in the second utterance
                (7) Elaboration: when the second utterance provides more information about the eventuality introduced in the first utterance
                (8) Explanation: when the second utterance explains the cause of what happened in the first utterance
                (9) Correction: when the second utterance corrects the first utterance
                (10) Contrast: when two utterances have similar semantic structures, but contrasting themes
                (11) Conditional: when the first utterance is a hypothesis and the second utterance is a consequence of the hypothesis
                (12) Background: when the second utterance provides some stage setting for the event that happened in the first utterance
                (13) Narration: when main eventualities of the first and second utterances occur in sequence
                (14) Alternation: when there is a disjunction between two utterances
                (15) Parallel: when two utterances have similar semantic structures and similar themes"""
    }

    user_message = {
        "role": "user",
        "content": f'Given the following dialog: \n{dialog} \nGenerate a question that is raised from the first utterance and answered by the second utterance. The first utterance is \'{utterance_1}\' and the second utterance is \'{utterance_2}\'. Please only respond with the question, without adding any additional information or context.'
    }

    return [system_message, user_message]

def get_follow_up_prompt_8(utterance_1, utterance_2):
    user_message = {
        "role": "user",
        "content": f'Now predict the relation between \'{utterance_1}\' and \'{utterance_2}\'? Please respond with one of the labels, without adding any additional information or context.'
    }

    return user_message