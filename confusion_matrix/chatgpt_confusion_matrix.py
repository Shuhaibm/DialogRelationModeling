import sys
sys.path.append('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/experiments_finetune/models')

import os
import argparse
from models.helpers import *
from prompts import *
from models.LlamaDataLoader import *
from models.MistralDataLoader import *

from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)


id2label = {0: "Comment", 1: "Clarification question", 2: "Question answer pair", 3: "Continuation",
                    4: "Acknowledgement", 5: "Question elaboration", 6: "Result", 7: "Elaboration", 8: "Explanation",
                    9: "Correction", 10: "Contrast", 11: "Conditional", 12: "Background", 13: "Narration",
                    14: "Alternation", 15: "Parallel", -1: "Invalid label"}
label2id = {v: k for k, v in id2label.items()}



def get_confusion_matrix(x_dev, y_dev):
    y_pred,y_true = [],[]
    for i,elem in enumerate(x_dev):
        x,y = elem,y_dev[i]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": x[0]["content"]},
                {"role": "user", "content": x[1]["content"]},
            ]
        )

        generated_text = response.choices[0].message.content
        y_pred.append(generated_text)
        y_true.append(y)

        print(f'\n\n\n***** Example #{i}')
        print(f'***** Model generated text: {generated_text}')
        print(f'***** Correct label {y}')

    f1_y_true,f1_y_pred = evaluate_performance(y_true, y_pred, label2id)

    # Get confusion matrix
    prediction_to_true_label = {} # Key = model prediction, value = list of the true answers when model predicted this
    for i,true in enumerate(f1_y_true):
        label,prediction = id2label[true],id2label[f1_y_pred[i]]
        
        if prediction not in prediction_to_true_label: prediction_to_true_label[prediction] = []
        prediction_to_true_label[prediction].append(label)

    return prediction_to_true_label

def test_model_with_confusion_matrix(x_test,y_test, prediction_to_true_label):
    y_pred,y_true = [],[]
    for i,elem in enumerate(x_test):
        x,y = elem,y_test[i]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": x[0]["content"]},
                {"role": "user", "content": x[1]["content"]},
            ]
        )

        generated_relation = response.choices[0].message.content
        y_pred.append(generated_relation)
        y_true.append(y)


        generated_label = -1
        for label in label2id:
            if label in generated_relation:
                generated_label = label
                break
        
        if generated_label == -1:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": x[0]["content"]},
                    {"role": "user", "content": x[1]["content"]},
                    {"role": "assistant", "content": generated_relation},
                    {"role": "user", "content": "You have not responded with one of the given labels. Please try again."},
                ]
            )
            generated_relation = response.choices[0].message.content
        else:
            if generated_label not in prediction_to_true_label:
                print("SKIPPING, SOMETHINGS WRONG")
                continue
            followup_message = f"You predicted {generated_relation}. "
            for relation in set(prediction_to_true_label[generated_label]):
                percent = 100*prediction_to_true_label[generated_label].count(relation)/len(prediction_to_true_label[generated_label])
                followup_message += f"{percent}% of the time when you predicted {generated_relation}, the correct answer was {relation}. "

            followup_message += "Considering this information, predict a label, and refrain from including any extra details or examples. "

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": x[0]["content"]},
                    {"role": "user", "content": x[1]["content"]},
                    {"role": "assistant", "content": generated_relation},
                    {"role": "user", "content": followup_message},
                ]
            )
            generated_relation = response.choices[0].message.content

        y_pred.append(generated_relation)
        y_true.append(y)

        print(f'\n\n\n***** Example #{i}')
        print(f'***** Model answer label {generated_relation}')
        print(f'***** Correct label {y}')

    f1_y_true,f1_y_pred = evaluate_performance(y_true, y_pred, label2id)

def main(prompt_fn):
    print("\nloading data")

    dev_stac = json.load(open('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/experiments_finetune/data/dialog_relation/dev_stac.json'))
    test_stac = json.load(open('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/experiments_finetune/data/dialog_relation/test_stac.json'))    
    x_dev,y_dev = [],[]
    for elem in dev_stac:
        dialog = elem["dialog"]
        utterance_1 = f'[Turn {elem["sentence_one"]["speechturn"]}] {elem["sentence_one"]["speaker"]}: {elem["sentence_one"]["text"]}'
        utterance_2 = f'[Turn {elem["sentence_two"]["speechturn"]}] {elem["sentence_two"]["speaker"]}: {elem["sentence_two"]["text"]}'
        relation = elem["relation"]
        curr_dialog = prompt_fn(dialog, utterance_1, utterance_2)

        x_dev.append(curr_dialog)
        y_dev.append(correct_relations[relation])
    
    x_test,y_test = [],[]
    for elem in test_stac:
        dialog = elem["dialog"]
        utterance_1 = f'[Turn {elem["sentence_one"]["speechturn"]}] {elem["sentence_one"]["speaker"]}: {elem["sentence_one"]["text"]}'
        utterance_2 = f'[Turn {elem["sentence_two"]["speechturn"]}] {elem["sentence_two"]["speaker"]}: {elem["sentence_two"]["text"]}'
        relation = elem["relation"]
        curr_dialog = prompt_fn(dialog, utterance_1, utterance_2)

        x_test.append(curr_dialog)
        y_test.append(correct_relations[relation])

    
    # Get confusion matrix
    print("Getting confusion matrix")
    prediction_to_true_label = get_confusion_matrix(x_dev,y_dev)

    # Test
    print("\n\n Test on test set, without confusion matrix")
    get_confusion_matrix(x_test, y_test)
    print("\n\n Test on test set, with confusion matrix")
    test_model_with_confusion_matrix(x_test, y_test, prediction_to_true_label)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=int, help="An integer number")
    args = parser.parse_args()

    if args.prompt == 3:
        prompt = get_prompt_3

    main(prompt)