from helpers import *

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