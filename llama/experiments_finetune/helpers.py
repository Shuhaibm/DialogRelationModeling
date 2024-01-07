import torch
from torch.nn.functional import softmax
import jsonlines
from sklearn.metrics import f1_score

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
    stac_train = jsonlines.open('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/data/stac/train_subindex.json')
    stac_dev = jsonlines.open('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/data/stac/dev_subindex.json')
    
    stac_dialogs = {}
    for dataset in [stac_test, stac_train, stac_dev]:
        for i,line in enumerate(dataset.iter()):
            curr_id = line["id"]
            context = ""
            for j,edu in enumerate(line["edus"]):
                edu_message = f'[Turn {j}] {edu["speaker"]}: {edu["text"]}'
                context += edu_message + "\n"
            
            stac_dialogs[curr_id] = context
    return stac_dialogs


