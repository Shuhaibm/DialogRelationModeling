from transformers import AutoTokenizer, DataCollatorWithPadding, LlamaForCausalLM, AutoModelForCausalLM, TrainingArguments, Trainer, set_seed
import torch
from optimum.bettertransformer import BetterTransformer
from peft import (get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP)
from sklearn.metrics import f1_score
import sys
sys.path.append('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/experiments_finetune/models')

import os
import argparse
from models.helpers import *
from prompts import *
from models.LlamaDataLoader import *
from models.MistralDataLoader import *


from transformers import logging as hf_logging
hf_logging.set_verbosity_debug()

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

    correct = sum([1 for i in range(total) if y_true[i] in y_pred[i]])
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

    print(f'accuracy: {correct/total}, total: {len(y_true)}, correct: {correct}')
    print(f"F1 Macro: {f1_macro}")
    print(f"F1 Micro: {f1_micro}")

    # Get confusino matrix
    prediction_to_true_label = {} # Key = model prediction, value = list of the true answers when model predicted this
    for i,true in enumerate(f1_y_true):
        label,prediction = id2label[true],id2label[f1_y_pred[i]]
        
        if prediction not in predictions_per_label: predictions_per_label[prediction] = []

        predictions_per_label[prediction].append(label)
    
    return prediction_to_true_label


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

        print("Dataloader")
        dataloader = LlamaDataLoader(tokenizer, prompt_fn, max_length)

        print("Quantizing")
        # quantization
        model = prepare_model_for_int8_training(model)
        print("PEFT")
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
        print("DONE")

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
    train_dataset,dev_dataset,test_dataset = dataloader.get_relation_data()
    tokenized_train_dataset,tokenized_dev_dataset = dataloader.tokenize_dataset(train_dataset),dataloader.tokenize_dataset(dev_dataset)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")


    # Fine tune
    print("\nfinetuning")
    training_args = TrainingArguments(
        output_dir="output_dir",
        overwrite_output_dir=True,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=10,
        per_device_train_batch_size=batch_size, #2, 3
        # learning_rate=learning_rate,#try 1e-4, 2e-4, 3e-4, 1e-5,2e-5,3e-5,4e-5,5e-5
        # num_train_epochs=epochs, #try 2-4
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    print("Done training")

    if model_type == "llama2": model.save_pretrained(f"./models/llama2/LlamaRelationPredictor")
    if model_type == "mistral": model.save_pretrained(f"./models/mistral/MistralRelationPredictor")

    # Get confusion matrix
    prediction_to_true_label = test_model(model, tokenizer, tokenized_train_dataset, max_length, dataloader.label2id, dataloader.id2label)
    print("prediction_to_true_label")
    print(prediction_to_true_label)
    for keys,values in prediction_to_true_label.items():
        print(keys)
        print(values)
    
    
    # Test
    print("Beginning test on dev set")
    test_model(model, tokenizer, tokenized_dev_dataset, max_length, dataloader.label2id, dataloader.id2label)
    

    
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