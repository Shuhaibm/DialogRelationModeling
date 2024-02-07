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

def test_model(model, tokenizer, test_dataset, max_length, label2id, id2label):
    print("\ntesting")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_pred,y_true = [],[]
    total = len(test_dataset)

    for i,elem in enumerate(test_dataset):
        x,y = elem["prompt"],elem["target"]

        model_input = tokenizer(x, return_tensors="pt").to(device)
        generations = model.generate(input_ids=model_input["input_ids"], max_new_tokens=max_length+500)
        generated_text = tokenizer.decode(generations[0], skip_special_tokens=True)

        answer = generated_text.split("\n")[-1]
        y_pred.append(answer)
        y_true.append(y)

        print(f'\n\n\n***** Example #{i}')
        print(f'***** Model generated text: {generated_text}')
        print(f'***** Model answer label {answer}')
        print(f'***** Correct label {y}')

def collect_questions(model, tokenizer, test_dataset, max_length):
    print("\nCollecting Questions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    question_dataset = test_dataset
    for i,elem in enumerate(test_dataset):
        x,y = elem["prompt"],elem["target"]

        model_input = tokenizer(x, return_tensors="pt").to(device)
        generations = model.generate(input_ids=model_input["input_ids"], max_new_tokens=max_length+500)
        generated_text = tokenizer.decode(generations[0], skip_special_tokens=True)

        generated_question = generated_text.split("[/INST] ")[-1]

        print(f'\n\n\n***** Example #{i}')
        print(f'***** Model generated text: {generated_text}')
        print(f'***** Model question: {generated_question}')
        print(f'***** Target question: {y}')

        question_dataset[i]["question"] = generated_question
        print(question_dataset[i])
    
    return question_dataset


def main(
    model_type,
    prompt_fn,
    max_length,
    batch_size,
    learning_rate,
    epochs
):
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load model and tokenizer
    print("\nloading model and tokenizer")
    if model_type == "llama2":
        base_model_dir,tokenizer_dir = "./models/llama2/LlamaForCausalLM","./models/llama2/tokenizer"
        if os.path.exists(base_model_dir) and os.path.exists(tokenizer_dir):
            model = LlamaForCausalLM.from_pretrained(base_model_dir, load_in_8bit=True, device_map="auto", use_cache=False)#.to(device)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            
            tokenizer.add_special_tokens({"pad_token":"<pad>"})
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id
            model.embed_tokens = torch.nn.Embedding(model.config.vocab_size, model.config.hidden_size, padding_idx=tokenizer.pad_token_id)

        else:
            tokenizer = AutoTokenizer.from_pretrained("./models/llama2/tokenizer")
            model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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", device_map="auto", use_cache=False)

            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = "right"

            model.save_pretrained(base_model_dir)
            tokenizer.save_pretrained(tokenizer_dir)

        dataloader = LlamaDataLoader(tokenizer, prompt_fn, max_length)

        # quantization
        model = prepare_model_for_int8_training(model)
        # PEFT
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules = ["q_proj", "v_proj"]
        )
        model = get_peft_model(model, peft_config)

    elif model_type == "mistral":
        # dataloader = MistralDataLoader(tokenizer, prompt_fn, max_length)

        base_model_dir,tokenizer_dir = "./models/mistral/model","./models/mistral/tokenizer" #TODO update /mistral/model once i have it downloaded
        if os.path.exists(base_model_dir) and os.path.exists(tokenizer_dir):
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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", device_map="auto", use_cache=False)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        else:
            model_id = "mistralai/Mistral-7B-Instruct-v0.2"
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.save_pretrained(tokenizer_dir)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
            model.save_pretrained(base_model_dir)

            return
        
        # TODO: add mistral configs if needed


    # Load data
    print("\nloading data")
    train_dataset,dev_dataset,test_dataset = dataloader.get_question_data()
    tokenized_train_dataset,tokenized_dev_dataset = dataloader.tokenize_dataset(train_dataset),dataloader.tokenize_dataset(dev_dataset)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # Collect Questions
    question_dataset = collect_questions(model, tokenizer, dev_dataset, max_length)
    question_dataset.to_json('dev_question_data_2.json', orient='records', lines=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--random_seed", type=int, help="An integer number")
    parser.add_argument("--prompt", type=int, help="An integer number")
    parser.add_argument("--max_length", type=int, help="An integer number")
    parser.add_argument("--batch_size", type=int, help="An integer number")
    parser.add_argument("--learning_rate", type=str)
    parser.add_argument("--epochs", type=int, help="An integer number")
    args = parser.parse_args()

    set_seed(args.random_seed)
    print(f'Random seed: {args.random_seed}\n')
    print(f'Model type: {args.model_type}\nPrompt: {args.prompt}\nMax length: {args.max_length}\nBatch size: {args.batch_size}\nLearning rate: {args.learning_rate}\nEpochs: {args.epochs}\n\n')

    if args.prompt == 10:
        prompt = get_question_prompt_10
    
    lr = float(args.learning_rate) if args.learning_rate else None
    main(args.model_type, prompt, args.max_length, args.batch_size, lr, args.epochs)