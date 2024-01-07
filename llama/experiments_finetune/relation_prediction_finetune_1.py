# from datasets import Dataset
# import evaluate
from transformers import AutoTokenizer, DataCollatorWithPadding, LlamaForCausalLM, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

import json
from relation_prediction_finetune_helpers import *
from dataset_helpers import *

def main(
    max_seq_len: int = 4096,
    max_batch_size: int = 8,
):
    train_stac = json.load(open('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/llama/experiments_finetune/data/train_stac.json'))
    dev_stac = json.load(open('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/llama/experiments_finetune/data/dev_stac.json'))
    test_stac = json.load(open('/home/shuhaibm/projects/def-vshwartz/shuhaibm/DialogRelationModeling/llama/experiments_finetune/data/test_stac.json'))
    x_train,y_train,id2label,label2id = prepare_dataset(train_stac, get_prompt_3)
    x_dev,y_dev,id2label,label2id = prepare_dataset(dev_stac, get_prompt_3)
    x_test,y_test,id2label,label2id = prepare_dataset(test_stac, get_prompt_3)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # model = LlamaForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-chat-hf", num_labels=len(label2id), id2label=id2label, label2id=label2id, token="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").bfloat16()
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


    # Prepare datasets


    # Fine tune
    #   - Use huggingface trainer

    # train_ds = Dataset.from_dict({"dialog": x, "relation": y})
    # accuracy = evaluate.load("accuracy")


    # def tokenize_function(examples):
    #     return tokenizer(examples["dialog"], padding="longest", max_length=max_seq_len, truncation=True)
    # tokenized_train_ds = train_ds.map(tokenize_function, batched=True)

    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    #Test
    # import evaluate    

    # accuracy = evaluate.load("accuracy")

    # def compute_metrics(eval_pred):
    #     predictions, labels = eval_pred
    #     predictions = np.argmax(predictions, axis=1)
    #     return accuracy.compute(predictions=predictions, references=labels)

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
    test_llama(model, tokenizer, x_test, y_test, id2label)

    
if __name__ == "__main__":
    main()