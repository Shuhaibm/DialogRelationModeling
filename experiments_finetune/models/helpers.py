import torch
from torch.nn.functional import softmax
import jsonlines
from sklearn.metrics import f1_score

correct_relations = {"Elaboration": "Elaboration",
                        "Explanation": "Explanation",
                        "Acknowledgement": "Acknowledgement",
                        "Question_answer_pair": "Question answer pair",
                        "Q_Elab": "Question elaboration",
                        "Clarification_question": "Clarification question",
                        "Comment": "Comment",
                        "Narration": "Narration",
                        "Continuation": "Continuation",
                        "Contrast": "Contrast",
                        "Parallel": "Parallel",
                        "Result": "Result",
                        "Background": "Background",
                        "Conditional": "Conditional",
                        "Alternation": "Alternation",
                        "Correction": "Correction"
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

def log_memory_usage(i):
    torch.cuda.empty_cache()

    print(f'Log memory: {i}')
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3) # Convert bytes to GB
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3) # Convert bytes to GB
    print(f"Allocated memory: {allocated_memory:.2f} GB")
    print(f"Reserved memory: {reserved_memory:.2f} GB")

def evaluate_performance(y_true, y_pred, label2id):
    correct = sum([1 for i,y_true_elem in enumerate(y_true) if y_true_elem in y_pred[i]])
    f1_y_true = [label2id[y_true_elem] for y_true_elem in y_true]
    f1_y_pred = []
    for y_pred_elem in y_pred:
        added = False
        for label in label2id:
            if label in y_pred_elem:
                f1_y_pred.append(label2id[label])
                added = True
                break
        if not added: f1_y_pred.append(-1)
    
    f1_macro,f1_micro = f1_score(f1_y_true, f1_y_pred, average='macro'), f1_score(f1_y_true, f1_y_pred, average='micro')

    print(f'accuracy: {correct/len(y_true)}, total: {len(y_true)}, correct: {correct}')
    print(f"F1 Macro: {f1_macro}")
    print(f"F1 Micro: {f1_micro}")

    return f1_y_true,f1_y_pred