import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from helpers import get_question_relation_data,get_sentence_pair_question_relation_data,get_masked_sentence_pair_question_relation_data,get_masked_sentence_pair_distance_question_relation_data
import random
random.seed(0)

def train_and_test_model(X_train,X_test,y_train,y_test):
    vectorizer = CountVectorizer()
    X_train_vector = vectorizer.fit_transform(X_train)
    X_test_vector = vectorizer.transform(X_test)

    classifier = MultinomialNB()
    classifier.fit(X_train_vector, y_train)

    y_pred = classifier.predict(X_test_vector)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

print("********** First run - input question**********\n")
X_train,X_test,y_train,y_test = get_question_relation_data()
train_and_test_model(X_train,X_test,y_train,y_test)


print("\n\n********** Second run - input sentence pairs and question**********\n")
X_train,X_test,y_train,y_test = get_sentence_pair_question_relation_data()
train_and_test_model(X_train,X_test,y_train,y_test)


print("\n\n********** Third run - masked speakers, input sentence pairs and question**********\n")
X_train,X_test,y_train,y_test = get_masked_sentence_pair_question_relation_data_for_lm()
train_and_test_model(X_train,X_test,y_train,y_test)


print("\n\n********** Fourth run - masked speakers, input sentence pairs, distance and question**********\n")
X_train,X_test,y_train,y_test = get_masked_sentence_pair_distance_question_relation_data()
train_and_test_model(X_train,X_test,y_train,y_test)