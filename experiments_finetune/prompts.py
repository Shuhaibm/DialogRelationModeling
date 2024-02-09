from models.helpers import *

system_message_1 = {
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

system_message_2 = {
        "role": "system", 
        "content": f"""Respond with the a question that aligns with one of the gold standard question types.
                (0) How is the content or situation in the previous statement being personally evaluated or commented on?
                (1) Can you provide more details or clarify what you just mentioned?
                (2) What is the response to the previously asked question?",
                (3) What additional information or details are provided on the same topic?
                (4) Is there an acknowledgment in response to the previous statement or situation?
                (5) What further clarification or information is being sought with the follow-up question?
                (6) What is the effect or result caused by the first statement?
                (7) How is the initial statement or situation further detailed or expanded upon?
                (8) What is the reason or cause for this situation?
                (9) Is the first statement being corrected or refuted as factual or accurate?
                (10) Is there a statement that contrasts or differs in theme or consequence from the first statement?
                (11) What will be the outcome if a certain condition is met?
                (12) What background or context information sets the stage for the main event or situation?
                (13) What happens next in the sequence of events?
                (14) What are the different options or alternatives presented?
                (15) Is there a statement that shares a common theme or idea with the first statement?

Please respond with only a question, without adding any additional information or context."""
    }

def get_prompt_3(dialog, utterance_1, utterance_2):
    user_message = {
        "role": "user",
        "content": f'Given the following dialog: \n{dialog} \nWhat is the relation between \'{utterance_1}\' and \'{utterance_2}\'? Please provide only the label that best fits your response, and refrain from including any extra details or examples.'
    }

    return [system_message_1, user_message]

def get_prompt_8(dialog, utterance_1, utterance_2):
    user_message = {
        "role": "user",
        "content": f'Given the following dialog: \n{dialog} \nGenerate a question that is raised from the first utterance and answered by the second utterance. The first utterance is \'{utterance_1}\' and the second utterance is \'{utterance_2}\'. Please only respond with the question, without adding any additional information or context.'
    }

    return [system_message_2, user_message]

def get_followup_prompt_8(dialog, utterance_1, utterance_2, question):
    user_message_1 = {
        "role": "user",
        "content": f'Given the following dialog: \n{dialog} \nGenerate a question that is raised from the first utterance and answered by the second utterance. The first utterance is \'{utterance_1}\' and the second utterance is \'{utterance_2}\'. Please only respond with the question, without adding any additional information or context.'
    }
    
    assistant_message = {"role": "assistant", "content": question}

    user_message_2 = {
        "role": "user",
        "content": f'Now predict the relation between \'{utterance_1}\' and \'{utterance_2}\'? Please respond with one of the labels, without adding any additional information or context.'
    }

    return [system_message_2, user_message_1, assistant_message, user_message_2]

def get_question_prompt_10(dialog, utterance_1, utterance_2):
    user_message = {
        "role": "user",
        "content": f'Given the following dialog: \n{dialog} \nGenerate a question that is raised from the first utterance and answered by the second utterance. The first utterance is \'{utterance_1}\' and the second utterance is \'{utterance_2}\'. Please only respond with the question, without adding any additional information or context.'
    }

    return [system_message_2, user_message]