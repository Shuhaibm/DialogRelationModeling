# Prepares input for question generation based on stac gold standard relations
import jsonlines
import random
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

random.seed(0)


# Load stac datasets
stac_dev = jsonlines.open('data/stac/dev_subindex.json')
stac_test = jsonlines.open('data/stac/test_subindex.json')
stac_train = jsonlines.open('data/stac/train_subindex.json')

# Select the dataset
curr_data_set = stac_dev


# Proess data into following format:
#   articles = [
#       {
#            article: [ ... ]
#            relations: [ ... ]
#       },
#       .
#       .
#       .
#   ]
articles = []
for i,line in enumerate(curr_data_set.iter()):
    curr_article = { "article": [], "relations": [] }
    for edu in line["edus"]:
        edu_context = f'{edu["speaker"]}: {edu["text"]}'
        curr_article["article"].append(edu_context)
    for relation in line["relations"]:
        x,y = relation["x"],relation["y"]
        curr_article["relations"].append([x, y])

    articles.append(curr_article)



tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)


fw=open('./ner.txt','w')
for curr_article in articles:
    sentences=[]
    sentences_after_ner=[]

    # TODO: Is NER necessary? I dont think so?
    for curr_sentence in curr_article["article"]:
        # ner_results = nlp(curr_sentence)
        sentences.append(curr_sentence)
        # curr_sentence=curr_sentence.split()

        # for item_entity in ner_results:
        #     print(item_entity)
        #     if not item_entity['word'][:2]=='##':
        #         item_entity_new=item_entity
        #     else:
        #         item_entity_new['word']=item_entity_new['word']+item_entity['word'][2:]

        # for item_entity in ner_results:
        #     if not item_entity['word'][:2]=='##':
        #         for word_num in range(len(curr_sentence)):
        #             if item_entity['word'] in curr_sentence[word_num]:
        #                 curr_sentence[word_num]=item_entity['entity'].split('-')[-1]

        # curr_sentence=' '.join(curr_sentence)
        # sentences_after_ner.append(curr_sentence)



    for relation in curr_article["relations"]:
        anchor_pos, answer_pos = min(relation), max(relation)

        # 1. Write every sentence before the current anchor sentence
        fw.write(' '.join(sentences[:anchor_pos]))
        # 2. Write "<@" + anchor sentence + "(>"
        fw.write(' <@ '+sentences[anchor_pos]+' (> ')
        # 3. Write every sentence after the anchor sentence up to the answer sentence (not including the answer sentence)
        fw.write(' '.join(sentences[anchor_pos+1:answer_pos]))
        # 4. Write "||" + anchor sentence + "||" + answer sentence
        fw.write(' || '+sentences[anchor_pos]+' || '+sentences[answer_pos])
        # 5. Write "|" + answer sentence (placeholder for question), then write a new line
        fw.write(' | '+sentences[answer_pos-1]+'\n') #placeholder for question TODO replace with a better placeholder after looking at question generation script


