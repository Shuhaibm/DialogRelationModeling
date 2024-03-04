from transformers import AutoTokenizer, DataCollatorWithPadding, LlamaForCausalLM, AutoModelForCausalLM, TrainingArguments, Trainer, set_seed
import torch
from optimum.bettertransformer import BetterTransformer
from peft import (get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP)
import sys
sys.path.append('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/experiments_finetune/models')

import os
import argparse
from models.helpers import *
from prompts import *
from models.LlamaDataLoader import *
from models.MistralDataLoader import *

def get_confusion_matrix(model, tokenizer, dev_dataset, max_length, label2id, id2label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_pred,y_true = [],[]
    for i,elem in enumerate(dev_dataset):
        x,y = elem["prompt"],elem["target"]

        model_input = tokenizer(x, return_tensors="pt").to(device)
        generations = model.generate(input_ids=model_input["input_ids"], max_new_tokens=max_length+500)
        generated_text = tokenizer.decode(generations[0], skip_special_tokens=True)
        generated_relation = generated_text.split("[/INST] ")[-1]

        y_pred.append(generated_relation)
        y_true.append(y)

        print(f'\n\n\n***** Example #{i}')
        print(f'***** Model generated text: {generated_text}')
        print(f'***** Model answer label {generated_relation}')
        print(f'***** Correct label {y}')

    f1_y_true,f1_y_pred = evaluate_performance(y_true, y_pred, label2id)

    # Get confusion matrix
    prediction_to_true_label = {} # Key = model prediction, value = list of the true answers when model predicted this
    for i,true in enumerate(f1_y_true):
        label,prediction = id2label[true],id2label[f1_y_pred[i]]
        
        if prediction not in prediction_to_true_label: prediction_to_true_label[prediction] = []
        prediction_to_true_label[prediction].append(label)

    return prediction_to_true_label

def test_model_with_confusion_matrix(model, tokenizer, test_dataset, max_length, prediction_to_true_label, label2id, id2label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_pred,y_true = [],[]
    for i,elem in enumerate(test_dataset):
        x,y = elem["prompt"],elem["target"]

        model_input = tokenizer(x, return_tensors="pt").to(device)
        generations = model.generate(input_ids=model_input["input_ids"], max_new_tokens=max_length+500)
        generated_text = tokenizer.decode(generations[0], skip_special_tokens=True)
        generated_relation = generated_text.split("[/INST] ")[-1]

        generated_label = -1
        for label in label2id:
            if label in generated_relation:
                generated_label = label
                break
        
        if generated_label == -1:
            followup_message = "You have not responded with one of the given labels. Please try again."
            followup_x = generated_text + " [INST] " + followup_message + " [/INST] "

            model_input = tokenizer(followup_x, return_tensors="pt").to(device)
            generations = model.generate(input_ids=model_input["input_ids"], max_new_tokens=max_length+500)
            generated_text = tokenizer.decode(generations[0], skip_special_tokens=True)
            generated_relation = generated_text.split("[/INST] ")[-1]
        else:
            if generated_label not in prediction_to_true_label:
                print("SKIPPING, SOMETHINGS WRONG")
                continue
            followup_message = f"You predicted {generated_relation}. "
            for relation in set(prediction_to_true_label[generated_label]):
                percent = 100*prediction_to_true_label[generated_label].count(relation)/len(prediction_to_true_label[generated_label])
                followup_message += f"{percent}% of the time when you predicted {generated_relation}, the correct answer was {relation}. "

            followup_message += "Considering this information, predict a label, and refrain from including any extra details or examples. "

            followup_x = generated_text + "[INST]" + followup_message + "[/INST]"

            model_input = tokenizer(followup_x, return_tensors="pt").to(device)
            generations = model.generate(input_ids=model_input["input_ids"], max_new_tokens=max_length+500)
            generated_text = tokenizer.decode(generations[0], skip_special_tokens=True)
            generated_relation = generated_text.split("[/INST] ")[-1]

        y_pred.append(generated_relation)
        y_true.append(y)

        print(f'\n\n\n***** Example #{i}')
        print(f'***** Model generated text: {generated_text}')
        print(f'***** Model answer label {generated_relation}')
        print(f'***** Correct label {y}')

    f1_y_true,f1_y_pred = evaluate_performance(y_true, y_pred, label2id)

def main(
    model_type,
    prompt_fn,
    max_length,
    batch_size,
):
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    print("\nloading model and tokenizer")
    if model_type == "llama2":
        base_model_dir,tokenizer_dir = "./models/llama2/LlamaForCausalLM","./models/llama2/tokenizer"

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        model = AutoModelForCausalLM.from_pretrained(base_model_dir, load_in_8bit=True, device_map="auto", use_cache=False)
        
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.embed_tokens = torch.nn.Embedding(model.config.vocab_size, model.config.hidden_size, padding_idx=tokenizer.pad_token_id)

        print("Dataloader")
        dataloader = LlamaDataLoader(tokenizer, prompt_fn, max_length)

        print("Quantizing")
        model = prepare_model_for_int8_training(model)
        print("PEFT")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules = ["q_proj", "v_proj"]
        )
        model = get_peft_model(model, peft_config)
 

    # Load data
    print("\nloading data")
    train_dataset,dev_dataset,test_dataset = dataloader.get_relation_data()
    tokenized_train_dataset,tokenized_dev_dataset = dataloader.tokenize_dataset(train_dataset),dataloader.tokenize_dataset(dev_dataset)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # Get confusion matrix
    prediction_to_true_label = get_confusion_matrix(model, tokenizer, dev_dataset, max_length, dataloader.label2id, dataloader.id2label)
    print("prediction_to_true_label")
    print(prediction_to_true_label)
    for keys,values in prediction_to_true_label.items():
        print(keys)
        print(values)

    # Test
    print("\n\n Test on test set, without confusion matrix")
    get_confusion_matrix(model, tokenizer, test_dataset, max_length, dataloader.label2id, dataloader.id2label)
    print("\n\n Test on test set, with confusion matrix")
    test_model_with_confusion_matrix(model, tokenizer, test_dataset, max_length, prediction_to_true_label, dataloader.label2id, dataloader.id2label)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--random_seed", type=int, help="An integer number")
    parser.add_argument("--prompt", type=int, help="An integer number")
    parser.add_argument("--max_length", type=int, help="An integer number")
    parser.add_argument("--batch_size", type=int, help="An integer number")
    args = parser.parse_args()

    set_seed(args.random_seed)
    print(f'Random seed: {args.random_seed}\n')
    print(f'Model type: {args.model_type}\nPrompt: {args.prompt}\nMax length: {args.max_length}\nBatch size: {args.batch_size}\n')

    if args.prompt == 3:
        prompt = get_prompt_3

    main(args.model_type, prompt, args.max_length, args.batch_size)