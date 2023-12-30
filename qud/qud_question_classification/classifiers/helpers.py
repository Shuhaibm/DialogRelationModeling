import torch
import json

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


dev_stac_qud = json.load(open('../data/dev_stac_qud.json'))
test_stac_qud = json.load(open('../data/test_stac_qud.json'))
train_stac_qud = json.load(open('../data/train_stac_qud.json'))

def get_question_relation_data():
    X_train,X_test,X_val,y_train,y_test,y_val = [],[],[],[],[],[]
    for elem in train_stac_qud:
        curr_x = elem['question']
        X_train.append(curr_x)
        curr_y = elem['relation']
        y_train.append(curr_y)

    for elem in dev_stac_qud:
        curr_x = elem['question']
        X_val.append(curr_x)
        curr_y = elem['relation']
        y_val.append(curr_y)

    for elem in test_stac_qud:
        curr_x = elem['question']
        X_test.append(curr_x)
        curr_y = elem['relation']
        y_test.append(curr_y)

    return X_train,X_test,y_train,y_test,X_val,y_val

def get_sentence_pair_relation_data(model_name):
    cls_token,sep_token = ""," "
    if model_name in ["bert", "todbert"]: 
        cls_token,sep_token = " [CLS] ", " [SEP] "
    elif model_name == "roberta": 
        cls_token,sep_token = " <s> ", " </s> "

    X_train,X_test,X_val,y_train,y_test,y_val = [],[],[],[],[],[]
    for elem in train_stac_qud:
        curr_x = cls_token + elem['sentence_one']['speaker'] + ": " + elem['sentence_one']['text'] + sep_token + elem['sentence_two']['speaker'] + ": " + elem['sentence_two']['text']
        X_train.append(curr_x)
        curr_y = elem['relation']
        y_train.append(curr_y)

    for elem in dev_stac_qud:
        curr_x = cls_token + elem['sentence_one']['speaker'] + ": " + elem['sentence_one']['text'] + sep_token + elem['sentence_two']['speaker'] + ": " + elem['sentence_two']['text']
        X_val.append(curr_x)
        curr_y = elem['relation']
        y_val.append(curr_y)

    for elem in test_stac_qud:
        curr_x = cls_token + elem['sentence_one']['speaker'] + ": " + elem['sentence_one']['text'] + sep_token + elem['sentence_two']['speaker'] + ": " + elem['sentence_two']['text']
        X_test.append(curr_x)
        curr_y = elem['relation']
        y_test.append(curr_y)

    return X_train,X_test,y_train,y_test,X_val,y_val


def get_sentence_pair_question_relation_data(model_name):
    cls_token,sep_token = ""," "
    if model_name in ["bert", "todbert"]: 
        cls_token,sep_token = " [CLS] ", " [SEP] "
    elif model_name == "roberta": 
        cls_token,sep_token = " <s> ", " </s> "

    X_train,X_test,X_val,y_train,y_test,y_val = [],[],[],[],[],[]
    for elem in train_stac_qud:
        curr_x = cls_token + elem['sentence_one']['speaker'] + ": " + elem['sentence_one']['text'] + sep_token + elem['sentence_two']['speaker'] + ": " + elem['sentence_two']['text'] + sep_token + elem['question']
        X_train.append(curr_x)
        curr_y = elem['relation']
        y_train.append(curr_y)

    for elem in dev_stac_qud:
        curr_x = cls_token + elem['sentence_one']['speaker'] + ": " + elem['sentence_one']['text'] + sep_token + elem['sentence_two']['speaker'] + ": " + elem['sentence_two']['text'] + sep_token + elem['question']
        X_val.append(curr_x)
        curr_y = elem['relation']
        y_val.append(curr_y)

    for elem in test_stac_qud:
        curr_x = cls_token + elem['sentence_one']['speaker'] + ": " + elem['sentence_one']['text'] + sep_token + elem['sentence_two']['speaker'] + ": " + elem['sentence_two']['text'] + sep_token + elem['question']
        X_test.append(curr_x)
        curr_y = elem['relation']
        y_test.append(curr_y)

    return X_train,X_test,y_train,y_test,X_val,y_val

def get_masked_sentence_pair_question_relation_data(model_name):
    cls_token,sep_token = ""," "
    if model_name in ["bert", "todbert"]: 
        cls_token,sep_token = " [CLS] ", " [SEP] "
    elif model_name == "roberta": 
        cls_token,sep_token = " <s> ", " </s> "

    X_train,X_test,X_val,y_train,y_test,y_val = [],[],[],[],[],[]
    for elem in train_stac_qud:
        curr_x = cls_token + elem['sentence_one']['speaker'] + ": " + elem['sentence_one']['text'] + sep_token + elem['sentence_two']['speaker'] + ": " + elem['sentence_two']['text'] + sep_token + elem['question']
        curr_x.replace(elem['sentence_one']['speaker'], "speaker_one")
        curr_x.replace(elem['sentence_two']['speaker'], "speaker_two")
        X_train.append(curr_x)
        curr_y = elem['relation']
        y_train.append(curr_y)

    for elem in dev_stac_qud:
        curr_x = cls_token + elem['sentence_one']['speaker'] + ": " + elem['sentence_one']['text'] + sep_token + elem['sentence_two']['speaker'] + ": " + elem['sentence_two']['text'] + sep_token + elem['question']
        curr_x = curr_x.replace(elem['sentence_one']['speaker'], "speaker_one")
        curr_x = curr_x.replace(elem['sentence_two']['speaker'], "speaker_two")
        X_val.append(curr_x)
        curr_y = elem['relation']
        y_val.append(curr_y)

    for elem in test_stac_qud:
        curr_x = cls_token + elem['sentence_one']['speaker'] + ": " + elem['sentence_one']['text'] + sep_token + elem['sentence_two']['speaker'] + ": " + elem['sentence_two']['text'] + sep_token + elem['question']
        curr_x = curr_x.replace(elem['sentence_one']['speaker'], "speaker_one")
        curr_x = curr_x.replace(elem['sentence_two']['speaker'], "speaker_two")
        X_test.append(curr_x)
        curr_y = elem['relation']
        y_test.append(curr_y)

    return X_train,X_test,y_train,y_test,X_val,y_val

def get_sentence_pair_distance_question_relation_data(model_name):
    cls_token,sep_token = ""," "
    if model_name in ["bert", "todbert"]: 
        cls_token,sep_token = " [CLS] ", " [SEP] "
    elif model_name == "roberta": 
        cls_token,sep_token = " <s> ", " </s> "

    X_train,X_test,X_val,y_train,y_test,y_val = [],[],[],[],[],[]
    for elem in train_stac_qud:
        curr_x = cls_token + str(elem['sentence_one']['speechturn']) + " speaker_one: " + elem['sentence_one']['text'] + sep_token + str(elem['sentence_two']['speechturn']) + " speaker_two: " + elem['sentence_two']['text'] + sep_token + elem['question']
        X_train.append(curr_x)
        curr_y = elem['relation']
        y_train.append(curr_y)

    for elem in dev_stac_qud:
        curr_x = cls_token + str(elem['sentence_one']['speechturn']) + " speaker_one: " + elem['sentence_one']['text'] + sep_token + str(elem['sentence_two']['speechturn']) + " speaker_two: " + elem['sentence_two']['text'] + sep_token + elem['question']
        X_val.append(curr_x)
        curr_y = elem['relation']
        y_val.append(curr_y)

    for elem in test_stac_qud:
        curr_x = cls_token + str(elem['sentence_one']['speechturn']) + " speaker_one: " + elem['sentence_one']['text'] + sep_token + str(elem['sentence_two']['speechturn']) + " speaker_two: " + elem['sentence_two']['text'] + sep_token + elem['question']
        X_test.append(curr_x)
        curr_y = elem['relation']
        y_test.append(curr_y)

    return X_train,X_test,y_train,y_test,X_val,y_val

def get_masked_sentence_pair_distance_question_relation_data(model_name):
    cls_token,sep_token = ""," "
    if model_name in ["bert", "todbert"]: 
        cls_token,sep_token = " [CLS] ", " [SEP] "
    elif model_name == "roberta": 
        cls_token,sep_token = " <s> ", " </s> "

    X_train,X_test,X_val,y_train,y_test,y_val = [],[],[],[],[],[]
    for elem in train_stac_qud:
        curr_x = cls_token + str(elem['sentence_one']['speechturn']) + " speaker_one: " + elem['sentence_one']['text'] + sep_token + str(elem['sentence_two']['speechturn']) + " speaker_two: " + elem['sentence_two']['text'] + sep_token + elem['question']
        curr_x = curr_x.replace(elem['sentence_one']['speaker'], "speaker_one")
        curr_x = curr_x.replace(elem['sentence_two']['speaker'], "speaker_two")
        X_train.append(curr_x)
        curr_y = elem['relation']
        y_train.append(curr_y)

    for elem in dev_stac_qud:
        curr_x = cls_token + str(elem['sentence_one']['speechturn']) + " speaker_one: " + elem['sentence_one']['text'] + sep_token + str(elem['sentence_two']['speechturn']) + " speaker_two: " + elem['sentence_two']['text'] + sep_token + elem['question']
        curr_x = curr_x.replace(elem['sentence_one']['speaker'], "speaker_one")
        curr_x = curr_x.replace(elem['sentence_two']['speaker'], "speaker_two")
        X_val.append(curr_x)
        curr_y = elem['relation']
        y_val.append(curr_y)

    for elem in test_stac_qud:
        curr_x = cls_token + str(elem['sentence_one']['speechturn']) + " speaker_one: " + elem['sentence_one']['text'] + sep_token + str(elem['sentence_two']['speechturn']) + " speaker_two: " + elem['sentence_two']['text'] + sep_token + elem['question']
        curr_x = curr_x.replace(elem['sentence_one']['speaker'], "speaker_one")
        curr_x = curr_x.replace(elem['sentence_two']['speaker'], "speaker_two")
        X_test.append(curr_x)
        curr_y = elem['relation']
        y_test.append(curr_y)

    return X_train,X_test,y_train,y_test,X_val,y_val


def get_sentence_pair_gibberish_question_relation_data(model_name):
    cls_token,sep_token = ""," "
    if model_name in ["bert", "todbert"]: 
        cls_token,sep_token = " [CLS] ", " [SEP] "
    elif model_name == "roberta": 
        cls_token,sep_token = " <s> ", " </s> "

    X_train,X_test,X_val,y_train,y_test,y_val = [],[],[],[],[],[]
    for elem in train_stac_qud:
        curr_x = cls_token + elem['sentence_one']['speaker'] + ": " + elem['sentence_one']['text'] + sep_token + elem['sentence_two']['speaker'] + ": " + elem['sentence_two']['text'] + sep_token + "hello test dog where sun?"
        X_train.append(curr_x)
        curr_y = elem['relation']
        y_train.append(curr_y)

    for elem in dev_stac_qud:
        curr_x = cls_token + elem['sentence_one']['speaker'] + ": " + elem['sentence_one']['text'] + sep_token + elem['sentence_two']['speaker'] + ": " + elem['sentence_two']['text'] + sep_token + "hello test dog where sun?"
        X_val.append(curr_x)
        curr_y = elem['relation']
        y_val.append(curr_y)

    for elem in test_stac_qud:
        curr_x = cls_token + elem['sentence_one']['speaker'] + ": " + elem['sentence_one']['text'] + sep_token + elem['sentence_two']['speaker'] + ": " + elem['sentence_two']['text'] + sep_token + "hello test dog where sun?"
        X_test.append(curr_x)
        curr_y = elem['relation']
        y_test.append(curr_y)

    return X_train,X_test,y_train,y_test,X_val,y_val

def get_sentence_pair_uniform_question_relation_data(model_name):
    cls_token,sep_token = ""," "
    if model_name in ["bert", "todbert"]: 
        cls_token,sep_token = " [CLS] ", " [SEP] "
    elif model_name == "roberta": 
        cls_token,sep_token = " <s> ", " </s> "

    X_train,X_test,X_val,y_train,y_test,y_val = [],[],[],[],[],[]
    for elem in train_stac_qud:
        curr_x = cls_token + elem['sentence_one']['speaker'] + ": " + elem['sentence_one']['text'] + sep_token + elem['sentence_two']['speaker'] + ": " + elem['sentence_two']['text'] + sep_token + "What is the response to the first message?"
        X_train.append(curr_x)
        curr_y = elem['relation']
        y_train.append(curr_y)

    for elem in dev_stac_qud:
        curr_x = cls_token + elem['sentence_one']['speaker'] + ": " + elem['sentence_one']['text'] + sep_token + elem['sentence_two']['speaker'] + ": " + elem['sentence_two']['text'] + sep_token + "What is the response to the first message?"
        X_val.append(curr_x)
        curr_y = elem['relation']
        y_val.append(curr_y)

    for elem in test_stac_qud:
        curr_x = cls_token + elem['sentence_one']['speaker'] + ": " + elem['sentence_one']['text'] + sep_token + elem['sentence_two']['speaker'] + ": " + elem['sentence_two']['text'] + sep_token + "What is the response to the first message?"
        X_test.append(curr_x)
        curr_y = elem['relation']
        y_test.append(curr_y)

    return X_train,X_test,y_train,y_test,X_val,y_val


def get_sentence_pair_relation_data_for_lm(tokenizer, model_name):
    labels = set([elem['relation'] for elem in train_stac_qud+test_stac_qud])
    label2id,id2label = {},{}
    for i,label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    X_train,X_test,y_train,y_test,X_val,y_val = get_sentence_pair_relation_data(model_name)

    padding = True
    
    train_encodings = tokenizer(X_train, truncation=True, padding=padding)
    val_encodings = tokenizer(X_val, truncation=True, padding=padding)
    test_encodings = tokenizer(X_test, truncation=True, padding=padding)

    train_labels = [label2id[label] for label in y_train]
    val_labels = [label2id[label] for label in y_val]
    test_labels = [label2id[label] for label in y_test]

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    return train_dataset,val_dataset,test_dataset,label2id,id2label,y_test

def get_question_relation_data_for_lm(tokenizer, model_name):
    labels = set([elem['relation'] for elem in train_stac_qud+test_stac_qud])
    label2id,id2label = {},{}
    for i,label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    X_train,X_test,y_train,y_test,X_val,y_val = get_question_relation_data()

    padding = True
    
    train_encodings = tokenizer(X_train, truncation=True, padding=padding)
    val_encodings = tokenizer(X_val, truncation=True, padding=padding)
    test_encodings = tokenizer(X_test, truncation=True, padding=padding)

    train_labels = [label2id[label] for label in y_train]
    val_labels = [label2id[label] for label in y_val]
    test_labels = [label2id[label] for label in y_test]

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    return train_dataset,val_dataset,test_dataset,label2id,id2label,y_test

def get_sentence_pair_question_relation_data_for_lm(tokenizer, model_name):
    labels = set([elem['relation'] for elem in train_stac_qud+test_stac_qud])
    label2id,id2label = {},{}
    for i,label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    X_train,X_test,y_train,y_test,X_val,y_val = get_sentence_pair_question_relation_data(model_name)

    padding = True
    
    train_encodings = tokenizer(X_train, truncation=True, padding=padding)
    val_encodings = tokenizer(X_val, truncation=True, padding=padding)
    test_encodings = tokenizer(X_test, truncation=True, padding=padding)

    train_labels = [label2id[label] for label in y_train]
    val_labels = [label2id[label] for label in y_val]
    test_labels = [label2id[label] for label in y_test]

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    return train_dataset,val_dataset,test_dataset,label2id,id2label,y_test

def get_masked_sentence_pair_question_relation_data_for_lm(tokenizer, model_name):
    labels = set([elem['relation'] for elem in train_stac_qud+test_stac_qud])
    label2id,id2label = {},{}
    for i,label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    X_train,X_test,y_train,y_test,X_val,y_val = get_masked_sentence_pair_question_relation_data(model_name)

    padding = True

    train_encodings = tokenizer(X_train, truncation=True, padding=padding)
    val_encodings = tokenizer(X_val, truncation=True, padding=padding)
    test_encodings = tokenizer(X_test, truncation=True, padding=padding)

    train_labels = [label2id[label] for label in y_train]
    val_labels = [label2id[label] for label in y_val]
    test_labels = [label2id[label] for label in y_test]

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    return train_dataset,val_dataset,test_dataset,label2id,id2label,y_test

def get_sentence_pair_distance_question_relation_data_for_lm(tokenizer, model_name):
    labels = set([elem['relation'] for elem in train_stac_qud+test_stac_qud])
    label2id,id2label = {},{}
    for i,label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    X_train,X_test,y_train,y_test,X_val,y_val = get_sentence_pair_distance_question_relation_data(model_name)

    padding = True
    
    train_encodings = tokenizer(X_train, truncation=True, padding=padding)
    val_encodings = tokenizer(X_val, truncation=True, padding=padding)
    test_encodings = tokenizer(X_test, truncation=True, padding=padding)

    train_labels = [label2id[label] for label in y_train]
    val_labels = [label2id[label] for label in y_val]
    test_labels = [label2id[label] for label in y_test]

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    return train_dataset,val_dataset,test_dataset,label2id,id2label,y_test

def get_masked_sentence_pair_distance_question_relation_data_for_lm(tokenizer, model_name):
    labels = set([elem['relation'] for elem in train_stac_qud+test_stac_qud])
    label2id,id2label = {},{}
    for i,label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    X_train,X_test,y_train,y_test,X_val,y_val = get_masked_sentence_pair_distance_question_relation_data(model_name)

    padding = True
    
    train_encodings = tokenizer(X_train, truncation=True, padding=padding)
    val_encodings = tokenizer(X_val, truncation=True, padding=padding)
    test_encodings = tokenizer(X_test, truncation=True, padding=padding)

    train_labels = [label2id[label] for label in y_train]
    val_labels = [label2id[label] for label in y_val]
    test_labels = [label2id[label] for label in y_test]

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    return train_dataset,val_dataset,test_dataset,label2id,id2label,y_test

def get_sentence_pair_gibberish_question_relation_data_for_lm(tokenizer, model_name):
    labels = set([elem['relation'] for elem in train_stac_qud+test_stac_qud])
    label2id,id2label = {},{}
    for i,label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    X_train,X_test,y_train,y_test,X_val,y_val = get_sentence_pair_gibberish_question_relation_data(model_name)

    padding = True
    
    train_encodings = tokenizer(X_train, truncation=True, padding=padding)
    val_encodings = tokenizer(X_val, truncation=True, padding=padding)
    test_encodings = tokenizer(X_test, truncation=True, padding=padding)

    train_labels = [label2id[label] for label in y_train]
    val_labels = [label2id[label] for label in y_val]
    test_labels = [label2id[label] for label in y_test]

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    return train_dataset,val_dataset,test_dataset,label2id,id2label,y_test

def get_sentence_pair_uniform_question_relation_data_for_lm(tokenizer, model_name):
    labels = set([elem['relation'] for elem in train_stac_qud+test_stac_qud])
    label2id,id2label = {},{}
    for i,label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    X_train,X_test,y_train,y_test,X_val,y_val = get_sentence_pair_uniform_question_relation_data(model_name)

    padding = True
    
    train_encodings = tokenizer(X_train, truncation=True, padding=padding)
    val_encodings = tokenizer(X_val, truncation=True, padding=padding)
    test_encodings = tokenizer(X_test, truncation=True, padding=padding)

    train_labels = [label2id[label] for label in y_train]
    val_labels = [label2id[label] for label in y_val]
    test_labels = [label2id[label] for label in y_test]

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    return train_dataset,val_dataset,test_dataset,label2id,id2label,y_test

