from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from helpers import *
import torch
import json
import sys
import random

if sys.argv[1] == "2e-5":
    lr = 2e-5
if sys.argv[1] == "2e-4":
    lr = 2e-4
if sys.argv[1] == "3e-4":
    lr = 3e-4

run = int(sys.argv[2])

def train_and_test_model(train_dataset,val_dataset,test_dataset,label2id,id2label,y_test,pad_token_id):
    random_seeds = [42,123,456]
    all_accuracy = []
    for random_seed in random_seeds:
        torch.cuda.empty_cache()
        random.seed(random_seed)
        print(f"Random seed = {random_seed}")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id), label2id=label2id, id2label=id2label)
        model.config.pad_token_id = pad_token_id
        model.resize_token_embeddings(len(tokenizer))

        training_args = TrainingArguments(
            output_dir=f'./{model_name}_sequence_classification_output',
            num_train_epochs=4,
            learning_rate=lr,
            save_total_limit=0,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        trainer.train()

        results = trainer.predict(test_dataset)
        test_predictions = results.predictions.argmax(axis=1)
        predicted_labels = [id2label[prediction] for prediction in test_predictions]


        accuracy = accuracy_score(y_test, predicted_labels)
        all_accuracy.append(accuracy)

        print("Accuracy:", accuracy)
        print(classification_report(y_test, predicted_labels))

        # model.save_pretrained(f'./{model_name}_sequence_classification_model')
        # tokenizer.save_pretrained(f'./{model_name}_sequence_classification_model')

    print("Average accuracy = " + str(sum(all_accuracy)/3))


model_name = "gpt2"
# model_dir = f"./{model_name}"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# train_dataset,val_dataset,test_dataset,label2id,id2label,y_test = get_masked_sentence_pair_question_relation_data_for_lm(tokenizer=tokenizer, model_name=model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id), label2id=label2id, id2label=id2label)
# model.resize_token_embeddings(len(tokenizer))
# tokenizer.save_pretrained(model_dir)
# model.save_pretrained(model_dir)

print(f"\n\n********** Run - model: {model_name}, learning rate = {lr} **********\n")

if run == 1:
    print("\n\n********** Run 1 - input sentence pairs **********\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    pad_token_id = tokenizer.pad_token_id
    train_dataset,val_dataset,test_dataset,label2id,id2label,y_test = get_sentence_pair_relation_data_for_lm(tokenizer=tokenizer, model_name=model_name)
    train_and_test_model(train_dataset,val_dataset,test_dataset,label2id,id2label,y_test,pad_token_id)
elif run == 2:
    print("\n\n********** Run 2 - input sentence pairs **********\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    pad_token_id = tokenizer.pad_token_id
    train_dataset,val_dataset,test_dataset,label2id,id2label,y_test = get_question_relation_data_for_lm(tokenizer=tokenizer, model_name=model_name)
    train_and_test_model(train_dataset,val_dataset,test_dataset,label2id,id2label,y_test,pad_token_id)
elif run == 3:
    print("\n\n********** Run 3 - input sentence pairs and question **********\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    pad_token_id = tokenizer.pad_token_id
    train_dataset,val_dataset,test_dataset,label2id,id2label,y_test = get_sentence_pair_question_relation_data_for_lm(tokenizer=tokenizer, model_name=model_name)
    train_and_test_model(train_dataset,val_dataset,test_dataset,label2id,id2label,y_test,pad_token_id)
elif run == 4:
    print("\n\n********** Run 4 - masked speakers, input sentence pairs and question **********\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    pad_token_id = tokenizer.pad_token_id
    train_dataset,val_dataset,test_dataset,label2id,id2label,y_test = get_masked_sentence_pair_question_relation_data_for_lm(tokenizer=tokenizer, model_name=model_name)
    train_and_test_model(train_dataset,val_dataset,test_dataset,label2id,id2label,y_test,pad_token_id)
elif run == 5:
    print("\n\n********** Run 5 - input sentence pairs, distance and question **********\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    pad_token_id = tokenizer.pad_token_id
    train_dataset,val_dataset,test_dataset,label2id,id2label,y_test = get_sentence_pair_distance_question_relation_data_for_lm(tokenizer=tokenizer, model_name=model_name)
    train_and_test_model(train_dataset,val_dataset,test_dataset,label2id,id2label,y_test,pad_token_id)
elif run == 6:
    print("\n\n********** Run 6 - masked speakers, input sentence pairs, distance and question **********\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    pad_token_id = tokenizer.pad_token_id
    train_dataset,val_dataset,test_dataset,label2id,id2label,y_test = get_masked_sentence_pair_distance_question_relation_data_for_lm(tokenizer=tokenizer, model_name=model_name)
    train_and_test_model(train_dataset,val_dataset,test_dataset,label2id,id2label,y_test,pad_token_id)
elif run == 7:
    print("\n\n********** Run 7 - input sentence pairs and place holder question (giberish) **********\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    pad_token_id = tokenizer.pad_token_id
    train_dataset,val_dataset,test_dataset,label2id,id2label,y_test = get_sentence_pair_gibberish_question_relation_data_for_lm(tokenizer=tokenizer, model_name=model_name)
    train_and_test_model(train_dataset,val_dataset,test_dataset,label2id,id2label,y_test,pad_token_id)
elif run == 8:
    print("\n\n********** Run 8 - input sentence pairs and uniform question **********\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    pad_token_id = tokenizer.pad_token_id
    train_dataset,val_dataset,test_dataset,label2id,id2label,y_test = get_sentence_pair_uniform_question_relation_data_for_lm(tokenizer=tokenizer, model_name=model_name)
    train_and_test_model(train_dataset,val_dataset,test_dataset,label2id,id2label,y_test,pad_token_id)

print(f"\n\n********** Run complete **********\n")