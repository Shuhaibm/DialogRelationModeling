from transformers import AutoTokenizer, DataCollatorWithPadding, LlamaForCausalLM, AutoModelForCausalLM, TrainingArguments, Trainer, set_seed
import torch
from optimum.bettertransformer import BetterTransformer
from peft import (get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP)
from sklearn.utils import shuffle

import sys
sys.path.append('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/experiments_finetune/models')

import os
import argparse
from models.helpers import *
from prompts import *
from models.LlamaDataLoader import *
from models.LlamaFollowupDataLoader import *
from models.MistralDataLoader import *

def test_model(model, tokenizer, dataset, max_length, label2id, id2label):
    print("\ntesting")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_pred,y_true = [],[]
    total = len(dataset)
    for i,elem in enumerate(dataset):
        x,y = elem["prompt"],elem["target"]

        model_input = tokenizer(x, return_tensors="pt").to(device)
        generations = model.generate(input_ids=model_input["input_ids"], max_new_tokens=max_length+500)
        generated_text = tokenizer.decode(generations[0], skip_special_tokens=True)

        generated_relation = generated_text.split("[/INST] ")[-1]

        y_pred.append(generated_relation)
        y_true.append(y)

        print(f'\n\n\n***** Example #{i}')
        print(f'***** Model generated text: {generated_text}')
        print(f'***** Model relation {generated_relation}')
        print(f'***** Target relation {y}')



    print(f'accuracy: {correct/total}, total: {len(y_true)}, correct: {correct}')
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



def main(
    model_type,
    prompt_fn,
    max_length,
    size
):
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load model and tokenizer
    print("\nloading model and tokenizer")
    if model_type == "llama2":
        model_dir = '/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/experiments_finetune/models/llama2/LlamaForCausalLM' if size==0 else f"./models/llama2/LlamaQuestionGenerator_{size}"
        tokenizer_dir = "./models/llama2/tokenizer"
        if os.path.exists(model_dir) and os.path.exists(tokenizer_dir):
            model = LlamaForCausalLM.from_pretrained(model_dir, load_in_8bit=True, device_map="auto", use_cache=False)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            
            tokenizer.add_special_tokens({"pad_token":"<pad>"})
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id
            model.embed_tokens = torch.nn.Embedding(model.config.vocab_size, model.config.hidden_size, padding_idx=tokenizer.pad_token_id)

        else:
            print(f"Model for size: {size} not found.")
            return
        
        dataset = f'/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/experiments_finetune/data/question_relation/dev_questions_finetune_{size}.json'
        if not os.path.exists(dataset):
            print(f"Dataset {dataset} not found")
            return
        dataloader = LlamaFollowupDataLoader(tokenizer, prompt_fn, max_length, dataset)

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

    # Load data
    print("\nloading data")
    dataset = dataloader.get_relation_data()

    # Prompt model
    test_model(model, tokenizer, dataset, max_length, dataloader.label2id, dataloader.id2label)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--random_seed", type=int, help="An integer number")
    parser.add_argument("--prompt", type=int, help="An integer number")
    parser.add_argument("--max_length", type=int, help="An integer number")
    parser.add_argument("--size", type=int, help="An integer number")
    args = parser.parse_args()

    set_seed(args.random_seed)
    print(f'Random seed: {args.random_seed}\n')
    print(f'Model type: {args.model_type}\nPrompt: {args.prompt}\nMax length: {args.max_length}\nSize (how many training examples the model is trained on): {args.size}\n\n')

    if args.prompt == 8:
        prompt = get_followup_prompt_8
    
    main(args.model_type, prompt, args.max_length, args.size)