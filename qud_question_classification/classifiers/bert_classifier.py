from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from helpers import *
import torch
import json
import sys

if sys.argv[1] == "2e-5":
    lr = 2e-5
if sys.argv[1] == "2e-4":
    lr = 2e-4
if sys.argv[1] == "3e-4":
    lr = 3e-4

def train_and_test_model(train_dataset,val_dataset,test_dataset,label2id,id2label,y_test):
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id), label2id=label2id, id2label=id2label)

    training_args = TrainingArguments(
        output_dir='./bert_sequence_classification_output',
        num_train_epochs=20,
        learning_rate=lr,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_total_limit=1,
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

    print("Accuracy:", accuracy)
    print(classification_report(y_test, predicted_labels))

    model.save_pretrained('./bert_sequence_classification_model')
    # tokenizer.save_pretrained('./bert_sequence_classification_model')



model_name = "bert-base-uncased"

print(f"\n\n********** Run - model: {model_name}, learning rate = {lr} **********\n")

print("\n\n********** First run - input sentence pairs and question**********\n")
tokenizer = BertTokenizer.from_pretrained(model_name)
train_dataset,val_dataset,test_dataset,label2id,id2label,y_test = get_sentence_pair_question_relation_data_for_lm(tokenizer=tokenizer)
train_and_test_model(train_dataset,val_dataset,test_dataset,label2id,id2label,y_test)

print("\n\n********** Second run - masked speakers, input sentence pairs and question**********\n")
tokenizer = BertTokenizer.from_pretrained(model_name)
train_dataset,val_dataset,test_dataset,label2id,id2label,y_test = get_masked_sentence_pair_question_relation_data_for_lm(tokenizer=tokenizer)
train_and_test_model(train_dataset,val_dataset,test_dataset,label2id,id2label,y_test)

print("\n\n********** Third run - masked speakers, input sentence pairs, distance and question**********\n")
tokenizer = BertTokenizer.from_pretrained(model_name)
train_dataset,val_dataset,test_dataset,label2id,id2label,y_test = get_masked_sentence_pair_distance_question_relation_data_for_lm(tokenizer=tokenizer)
train_and_test_model(train_dataset,val_dataset,test_dataset,label2id,id2label,y_test)

print(f"\n\n********** Run complete **********\n")