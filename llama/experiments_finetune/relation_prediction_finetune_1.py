from transformers import AutoTokenizer, DataCollatorWithPadding, LlamaForCausalLM, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

import json
from helpers import *
from prompts import *
from DataLoader import *

def test_llama(model, tokenizer, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_pred,y_true = [],[]
    total = len(test_dataset)

    for i,elem in enumerate(test_dataset):
        x,y = elem["feature"],elem["label"]

        model_input = tokenizer(x, return_tensors="pt").to(device)

        generations = model.generate(model_input["input_ids"], max_length=4096)
        generated_text = tokenizer.decode(generations[0], skip_special_tokens=True)

        answer = generated_text.split("\n")[-1]
        y_pred.append(answer)
        y_true.append(y)

        print(f'\n\n\n***** Example #{i}')
        print(f'***** Model generated text: {generated_text}')
        print(f'***** Model answer label {answer}')
        print(f'***** Correct label {y}')

    correct = sum([1 for i in range(total) if y_true[i] in y_pred[i]])
    f1_macro,f1_micro = f1_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred, average='micro')

    print(f'accuracy: {correct/total}, total: {len(y_true)}, correct: {correct}')
    print(f"F1 Macro: {f1_macro}")
    print(f"F1 Micro: {f1_micro}")

def main(
    max_seq_len: int = 4096,
    max_batch_size: int = 8,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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").bfloat16()
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")    
    model = LlamaForCausalLM.from_pretrained("./model/LlamaForCausalLM").bfloat16()
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")

    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.embed_tokens = torch.nn.Embedding(model.config.vocab_size, model.config.hidden_size, padding_idx=tokenizer.pad_token_id)

    # model.save_pretrained("./model/LlamaForCausalLM")


    dataloader = DataLoader(tokenizer, get_prompt_3)
    train_dataset,dev_dataset,test_dataset = dataloader.get_data()


    # Fine tune


    # TODO: prepare dataset to pass it to the huggingface trainer!
    # Begine finetuning, look at hyper params and stuff...


    # training_args = TrainingArguments(
    #     output_dir="clf",
    #     learning_rate=learning_rate,
    #     per_device_train_batch_size=batch_size,
    #     per_device_eval_batch_size=batch_size,
    #     num_train_epochs=epochs,
    #     weight_decay=0.01,
    #     evaluation_strategy="epoch",
    #     save_strategy="no",
    #     load_best_model_at_end=False,
    #     push_to_hub=False,
    # )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_ds["train"],
    #     eval_dataset=tokenized_ds[test_name],
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    # )
    
    # Test
    test_llama(model, tokenizer, test_dataset)

    
if __name__ == "__main__":
    main()