import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import json
from sklearn.metrics import accuracy_score, classification_report

from helpers import *
import random
random.seed(0)


# Hyperparameters
num_epochs = 10


class RelationClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(RelationClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)
    

def train_and_test_model(X_train,X_test,y_train,y_test):
    vectorizer = TfidfVectorizer()
    X_train,X_test = vectorizer.fit_transform(X_train),vectorizer.transform(X_test)

    label_to_id = {label: i for i, label in enumerate(set(y_train+y_test))}
    y_train = torch.tensor([label_to_id[label] for label in y_train], dtype=torch.long)
    y_test = torch.tensor([label_to_id[label] for label in y_test], dtype=torch.long)


    model = RelationClassifier(X_train.shape[1], len(label_to_id))
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train.toarray(), dtype=torch.float32))
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()


    # Testing the model
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X_test.toarray(), dtype=torch.float32))
        _, predicted = torch.max(outputs, 1)

    # Convert numerical labels back to text labels
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    predicted_labels = [id_to_label[idx.item()] for idx in predicted]

    # Evaluate the model
    accuracy = accuracy_score(y_test, predicted)
    report = classification_report(y_test, predicted)

    print("id_to_label: ", id_to_label)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)



print("********** First run - input question**********\n")

X_train,X_test,y_train,y_test = get_question_relation_data()
train_and_test_model(X_train,X_test,y_train,y_test)


print("\n\n********** Second run - input sentence pairs and question**********\n")
X_train,X_test,y_train,y_test = get_sentence_pair_question_relation_data()
train_and_test_model(X_train,X_test,y_train,y_test)


print("\n\n********** Third run - masked speakers, input sentence pairs and question**********\n")
X_train,X_test,y_train,y_test = get_masked_sentence_pair_question_relation_data()
train_and_test_model(X_train,X_test,y_train,y_test)

print("\n\n********** Fourth run - masked speakers, input sentence pairs, distance and question**********\n")
X_train,X_test,y_train,y_test = get_masked_sentence_pair_distance_question_relation_data()
train_and_test_model(X_train,X_test,y_train,y_test)