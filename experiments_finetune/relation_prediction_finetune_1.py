from transformers import AutoTokenizer, DataCollatorWithPadding, LlamaForCausalLM, AutoModelForCausalLM, TrainingArguments, Trainer, set_seed
import torch
from optimum.bettertransformer import BetterTransformer
from peft import (get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP)

import os
import json
import argparse
from helpers import *
from prompts import *
from LlamaDataLoader import *

def test_llama(model, tokenizer, test_dataset, max_length, label2id, id2label):
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
        for id in id2label:
            if id in y_pred_elem: f1_y_pred.append(id2label[id])
    
    [label2id[y_pred_elem] for y_pred_elem in y_pred]
    f1_macro,f1_micro = f1_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred, average='micro')

    print(f'accuracy: {correct/total}, total: {len(y_true)}, correct: {correct}')
    print(f"F1 Macro: {f1_macro}")
    print(f"F1 Micro: {f1_micro}")

def main(
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
    base_model_dir,tokenizer_dir = "./model/LlamaForCausalLM","./tokenizer"
    if os.path.exists(base_model_dir) and os.path.exists(tokenizer_dir):
        model = LlamaForCausalLM.from_pretrained(base_model_dir, load_in_8bit=True, device_map="auto", use_cache=False)#.to(device)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.embed_tokens = torch.nn.Embedding(model.config.vocab_size, model.config.hidden_size, padding_idx=tokenizer.pad_token_id)

    else:
        tokenizer = AutoTokenizer.from_pretrained("./tokenizer2")
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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", device_map="auto", use_cache=False)

        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

        model.save_pretrained(base_model_dir)
        tokenizer.save_pretrained(tokenizer_dir)


    # # use_fast_kernels (can't do beause I cant upgrade torch to 2.1.1)
    # model = BetterTransformer.transform(model)

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
    model.print_trainable_parameters()

    # FSDP
    # from torch.distributed.fsdp import ShardingStrategy
    # from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    # fsdp_config = {
    #     mixed_precision: True,
    #     use_fp16: False,
    #     sharding_strategy: ShardingStrategy.FULL_SHARD
    #     checkpoint_type: StateDictType.SHARDED_STATE_DICT,
    #     fsdp_activation_checkpointing: True,
    #     fsdp_cpu_offload: False,
    #     pure_bf16: False,
    #     optimizer: "AdamW"
    # }

    # mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
    # my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
    # torch.cuda.set_device(num_gpus)
    # model = FSDP(
        # model,
        # auto_wrap_policy= my_auto_wrapping_policy,
        # mixed_precision=mixed_precision_policy,
        # sharding_strategy=fsdp_config.sharding_strategy,
        # limit_all_gathers=True,
        # sync_module_states=train_config.low_cpu_fsdp,
        # param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
        # if train_config.low_cpu_fsdp and rank != 0 else None,
    # )
    # if fsdp_config.fsdp_activation_checkpointing:
        # apply_fsdp_checkpointing(model)


    # Load data
    print("\nloading data")
    dataloader = LlamaDataLoader(tokenizer, prompt_fn, max_length)
    train_dataset,dev_dataset,test_dataset = dataloader.get_data()
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
        learning_rate=learning_rate,#try 1e-4, 2e-4, 3e-4, 1e-5,2e-5,3e-5,4e-5,5e-5
        num_train_epochs=epochs, #try 2-4
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset[:1000], #use a subset for hyperparameter tuning
        data_collator=data_collator,
    )

    trainer.train()

    # Test
    test_llama(model, tokenizer, dev_dataset, max_length, dataloader.label2id, dataloader.id2label )

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, help="An integer number")
    parser.add_argument("--prompt", type=int, help="An integer number")
    parser.add_argument("--max_length", type=int, help="An integer number")
    parser.add_argument("--batch_size", type=int, help="An integer number")
    parser.add_argument("--learning_rate", type=str, help="An integer number")
    parser.add_argument("--epochs", type=int, help="An integer number")
    args = parser.parse_args()

    set_seed(args.random_seed)
    print(f'Random seed: {args.random_seed}\n')
    print(f'Prompt: {args.prompt}\nMax length: {args.max_length}\nBatch size: {args.batch_size}\nLearning rate: {args.learning_rate}\nEpochs: {args.epochs}\n\n')

    if args.prompt == 3:
        prompt = get_prompt_3

    main(prompt, args.max_length, args.batch_size, float(args.learning_rate), args.epochs)