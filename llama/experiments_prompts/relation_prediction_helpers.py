import jsonlines
import json

correct_relations = {"Elaboration": "(7) Elaboration",
                        "Explanation": "(8) Explanation",
                        "Acknowledgement": "(4) Acknowledgement",
                        "Question_answer_pair": "(2) Question answer pair",
                        "Q_Elab": "(5) Question elaboration",
                        "Clarification_question": "(1) Clarification question",
                        "Comment": "(0) Comment",
                        "Narration": "(13) Narration",
                        "Continuation": "(3) Continuation",
                        "Contrast": "(10) Contrast",
                        "Parallel": "(15) Parallel",
                        "Result": "(6) Result",
                        "Background": "(12) Background",
                        "Conditional": "(11) Conditional",
                        "Alternation": "(14) Alternation",
                        "Correction": "(9) Correction"
                        }

relation_to_gpt4_gold_standard_question = {"Elaboration": "How is the initial statement or situation further detailed or expanded upon?",
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

relation_to_gold_standard_question = {
                          "Elaboration": "Is there more information about the eventuality introduced in the first utterance?",
                          "Explanation": "Is there any explanation to what happened in the first utterance?",
                          "Acknowledgement": "Is there any signal of understanding or acceptance of the first utterance?",
                          "Question_answer_pair": "What is the answer to the question given in the first utterance?",
                          "Q_Elab": "Is there a follow-up question for the question presented in the first utterance?",
                          "Clarification_question": "Is there any clarification on what was said in the furst utterance?",
                          "Comment": "Is there an opinion or evaluation of the first utterance?",
                          "Narration": "Is there an eventuality that happens in sequence to the eventuality of the first utterance?",
                          "Continuation": "Is there any more background to the segment from the first utterance?",
                          "Contrast": "Is there an utterance that has a similar semantic structure, but a contrasting theme?",
                          "Parallel": "Is there an utterance that has a similar semantic structure, and a similar theme?",
                          "Result": "Is there anything that was caused by the main eventuality of the first utterance?",
                          "Background": "Is there some stage setting for the event that happened in the first utterance?",
                          "Conditional": "Is there an utterance that is a consequence of the hypothesis presented in the first utterance?",
                          "Alternation": "Is there a disjunction to the first utterance?",
                          "Correction": "Is there any correction to the first utterance"}

def get_stac_dialogs():
    stac_test = jsonlines.open('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/data/stac/test_subindex.json')
    stac_dialogs = {}
    for i,line in enumerate(stac_test.iter()):
        curr_id = line["id"]
        context = ""
        for j,edu in enumerate(line["edus"]):
            edu_message = f'[Turn {j}] {edu["speaker"]}: {edu["text"]}'
            context += edu_message + "\n"
        
        stac_dialogs[curr_id] = context
    return stac_dialogs

def get_prompt_1(dialog, utterance_1, utterance_2): # labels, dialog, utterances
    system_message = {
        "role": "system", 
        "content": f"""Respond with only one of the following labels, without adding any additional information or context:
                (0) Comment
                (1) Clarification question
                (2) Question answer pair
                (3) Continuation
                (4) Acknowledgement
                (5) Question elaboration
                (6) Result
                (7) Elaboration
                (8) Explanation
                (9) Correction
                (10) Contrast
                (11) Conditional
                (12) Background
                (13) Narration
                (14) Alternation
                (15) Parallel

                Please provide only the label that best fits your response, and refrain from including any extra details or examples.
                """
    }

    user_message = {
        "role": "user",
        "content": f'Given the following dialog: \n{dialog} \nWhat is the relation between \'{utterance_1}\' and \'{utterance_2}\'?'
    }

    return [system_message, user_message]
    
def get_prompt_2(dialog, utterance_1, utterance_2): # labels, dialog, utterances
    system_message = {
        "role": "system", 
        "content": f"""Respond with only one of the following labels, without adding any additional information or context:
                (0) Comment
                (1) Clarification question
                (2) Question answer pair
                (3) Continuation
                (4) Acknowledgement
                (5) Question elaboration
                (6) Result
                (7) Elaboration
                (8) Explanation
                (9) Correction
                (10) Contrast
                (11) Conditional
                (12) Background
                (13) Narration
                (14) Alternation
                (15) Parallel
                """
    }

    user_message = {
        "role": "user",
        "content": f'Given the following dialog: \n{dialog} \nWhat is the relation between \'{utterance_1}\' and \'{utterance_2}\'? Please provide only the label that best fits your response, and refrain from including any extra details or examples.'
    }

    return [system_message, user_message]
    
def get_prompt_3(dialog, utterance_1, utterance_2): # labels, definitions, dialog, utterances
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
    
def get_prompt_4(dialog, utterance_1, utterance_2, question): # labels, definitinos, dialog, utterances, DCQA qud-questions
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
        "content": f'Given the following dialog: \n{dialog} \nGenerate a question that is raised from the first utterance, \'{utterance_1}\', and answered by the second utterance, \'{utterance_2}\'.'
    }

    assistant_message = {
        "role": "assistant",
        "content": question
    }

    user_message_2 = {
        "role": "user",
        "content": f'Now predict the relation between the two utterances. Make sure to refer to the definitions. Please provide only the label that best fits your response, and refrain from including any extra details or examples.'
    }

    return [system_message, user_message, assistant_message, user_message_2]

def get_prompt_5(dialog, utterance_1, utterance_2, relation): # labels, definitinos, dialog, utterances, gold standard qud-questions --oracle
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
        "content": f'Given the following dialog: \n{dialog} \nGenerate a question that is raised from the first utterance, \'{utterance_1}\', and answered by the second utterance, \'{utterance_2}\'.'
    }

    assistant_message = {
        "role": "assistant",
        "content": relation_to_gpt4_gold_standard_question[relation]
    }

    user_message_2 = {
        "role": "user",
        "content": f'Now use this question, along with the provided definitions, to predict the relation between the two utterances. Please provide only the label that best fits your response, and refrain from including any extra details or examples.'
    }

    return [system_message, user_message, assistant_message, user_message_2]

def get_prompt_5_2(dialog, utterance_1, utterance_2, relation): # labels, definitions, dialog, utterances, gold standard qud-questions --oracle
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
        "content": f'Given the following dialog: \n{dialog} \nGenerate a question that is raised from the first utterance, \'{utterance_1}\', and answered by the second utterance, \'{utterance_2}\'.'
    }

    assistant_message = {
        "role": "assistant",
        "content": relation_to_gpt4_gold_standard_question[relation]
    }

    user_message_2 = {
        "role": "user",
        "content": f'Now predict the relation between the two utterances. Please provide only the label that best fits your response, and refrain from including any extra details or examples.'
    }

    return [system_message, user_message, assistant_message, user_message_2]

def get_prompt_5_3(dialog, utterance_1, utterance_2, relation): # labels, definitinos, dialog, utterances, gold standard qud-questions --oracle
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
        "content": f'Given the following dialog: \n{dialog} \nGenerate a question that is raised from the first utterance, \'{utterance_1}\', and answered by the second utterance, \'{utterance_2}\'.'
    }

    assistant_message = {
        "role": "assistant",
        "content": relation_to_gold_standard_question[relation]
    }

    user_message_2 = {
        "role": "user",
        "content": f'Now predict the relation between the two utterances. Please provide only the label that best fits your response, and refrain from including any extra details or examples.'
    }

    return [system_message, user_message, assistant_message, user_message_2]


def get_prompt_6(dialog, utterance_1, utterance_2, relation): # labels, definitinos, dialog, utterances, relation --oracle
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
        "content": f'Given the following dialog: \n{dialog} \n The relation between the two utterances is {relation}. Predict the relation between the two utterances. Please provide only the label that best fits your response, and refrain from including any extra details or examples.'
    }

    return [system_message, user_message]

def get_prompt_7(dialog, utterance_1, utterance_2): # labels, definitions, dialog, utterances
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
        "content": f'Given the following dialog: \n{dialog} \nDetermine the definition best defines the two utterances, \'{utterance_1}\' and \'{utterance_2}\'. What is the relation between them?'
    }

    return [system_message, user_message]

def get_prompt_7_2(dialog, utterance_1, utterance_2): # labels, definitions, dialog, utterances
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
        "content": f'Given the following dialog: \n{dialog} \nWhat definition best defines the relation between the two utterances, \'{utterance_1}\' and \'{utterance_2}\'?'
    }

    return [system_message, user_message]

def get_follow_up_prompt_7_2(utterance_1, utterance_2):
    user_message = {
        "role": "user",
        "content": f'Using that definition and the given labels, tell me the relation between \'{utterance_1}\' and \'{utterance_2}\'? Please respond with one of the labels, without adding any additional information or context.'
    }

    return user_message

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

def get_prompt_9(dialog, utterance_1, utterance_2): # labels, definitions, dialog, utterances
    system_message = {
        "role": "system", 
        "content": f"""The labels for every relation are listed below, along with an example of two utterances that have that relation:
                (0) Comment: '[Turn 20] yiin: finally' and '[Turn 21] Gaeilgeoir: haha'
                (1) Clarification question: '[Turn 10] yiin: thank you' and '[Turn 11] nareik15: so was it wood you were after?'
                (2) Question answer pair: '[Turn 0] Gaeilgeoir: anyone have clay? :)' and '[Turn 2] yiin: no sorry'
                (3) Continuation: '[Turn 13] Gaeilgeoir: I'll try again' and '[Turn 15] Gaeilgeoir: I'm definitely giving wheat'
                (4) Acknowledgement: '[Turn 0] Gaeilgeoir: well played' and '[Turn 1] inca: cheers, good game'
                (5) Question elaboration: '[Turn 4] inca: shall we say wednesday for the one without kieran?' and '[Turn 6] Gaeilgeoir: time?'
                (6) Result: '[Turn 10] Markus: Ah great!' and '[Turn 11] Markus: So, one of you can now start the game.'
                (7) Elaboration: '[Turn 1] william: does someone need clay?' and '[Turn 2] william: i will take sheep or ore'
                (8) Explanation: '[Turn 13] Thomas: retry' and '[Turn 14] Thomas: couldn't see it'
                (9) Correction: '[Turn 9] Shawnus: and that biotes me in the arse' and '[Turn 10] Shawnus: bites*'
                (10) Contrast: '[Turn 22] ztime: i did..' and '[Turn 23] ztime: but just traded with bank...'
                (11) Conditional: '[Turn 27] ztime: unless it's wood..' and '[Turn 28] ztime: I can't help'
                (12) Background: '[Turn 0] Cat: so I'd like to build a road...' and '[Turn 11] william: you need 10 victory points to win. '
                (13) Narration: '[Turn 1] william: first we chat about it' and '[Turn 2] william: when we actually want to exchange...'
                (14) Alternation: '[Turn 1] Cat: I'd like clay' and '[Turn 4] Cat: or sheep'
                (15) Parallel: '[Turn 1] inca: none at all,' and '[Turn 3] CheshireCatGrin: likewise'"""
    }

    user_message = {
        "role": "user",
        "content": f'Given the following dialog: \n{dialog} \nWhat is the relation between \'{utterance_1}\' and \'{utterance_2}\'? Please provide only the label that best fits your response, and refrain from including any extra details or examples.'
    }

    return [system_message, user_message]

def get_prompt_10(dialog, utterance_1, utterance_2): # labels, definitions, dialog, utterances
    system_message = {
        "role": "system", 
        "content": f"""The labels for every relation are listed below, along with definitions and an example of two utterances that have that relation:
                (0) Comment: when the second utterance provides an opinion or evaluation of the first utterance. Example:'[Turn 20] yiin: finally' and '[Turn 21] Gaeilgeoir: haha'
                (1) Clarification question: Clarification question: when the second utterance tries to clarify what was said in the first utterance. Example: '[Turn 10] yiin: thank you' and '[Turn 11] nareik15: so was it wood you were after?'
                (2) Question answer pair: when the second utterance gives an answer to a question in the first utterance. Example: '[Turn 0] Gaeilgeoir: anyone have clay? :)' and '[Turn 2] yiin: no sorry'
                (3) Continuation: when the first and second utterances elaborate or provide background to the same segment. Example: '[Turn 13] Gaeilgeoir: I'll try again' and '[Turn 15] Gaeilgeoir: I'm definitely giving wheat'
                (4) Acknowledgement: when the second utterance signals an understanding or acceptance of the first utterance. Example: '[Turn 0] Gaeilgeoir: well played' and '[Turn 1] inca: cheers, good game'
                (5) Question elaboration: when the second utterance is a follow-up question intended to get more information to answer the question presented in the first utterance. Example:'[Turn 4] inca: shall we say wednesday for the one without kieran?' and '[Turn 6] Gaeilgeoir: time?'
                (6) Result: when the main eventuality of the first utterance is understood to cause the eventuality given in the second utterance. Example: '[Turn 10] Markus: Ah great!' and '[Turn 11] Markus: So, one of you can now start the game.'
                (7) Elaboration: when the second utterance provides more information about the eventuality introduced in the first utterance. Example:'[Turn 1] william: does someone need clay?' and '[Turn 2] william: i will take sheep or ore'
                (8) Explanation: when the second utterance explains the cause of what happened in the first utterance. Example: '[Turn 13] Thomas: retry' and '[Turn 14] Thomas: couldn't see it'
                (9) Correction: when the second utterance corrects the first utterance. Example: '[Turn 9] Shawnus: and that biotes me in the arse' and '[Turn 10] Shawnus: bites*'
                (10) Contrast: when two utterances have similar semantic structures, but contrasting themes. Example: '[Turn 22] ztime: i did..' and '[Turn 23] ztime: but just traded with bank...'
                (11) Conditional: when the first utterance is a hypothesis and the second utterance is a consequence of the hypothesis. Example: '[Turn 27] ztime: unless it's wood..' and '[Turn 28] ztime: I can't help'
                (12) Background: when the second utterance provides some stage setting for the event that happened in the first utterance. Example: '[Turn 0] Cat: so I'd like to build a road...' and '[Turn 11] william: you need 10 victory points to win. '
                (13) Narration: when main eventualities of the first and second utterances occur in sequence. Example: '[Turn 1] william: first we chat about it' and '[Turn 2] william: when we actually want to exchange...'
                (14) Alternation: when there is a disjunction between two utterances. Example: '[Turn 1] Cat: I'd like clay' and '[Turn 4] Cat: or sheep'
                (15) Parallel: when two utterances have similar semantic structures and similar themes. Example: '[Turn 1] inca: none at all,' and '[Turn 3] CheshireCatGrin: likewise'"""
    }

    user_message = {
        "role": "user",
        "content": f'Given the following dialog: \n{dialog} \nWhat is the relation between \'{utterance_1}\' and \'{utterance_2}\'? Please provide only the label that best fits your response, and refrain from including any extra details or examples.'
    }

    return [system_message, user_message]

