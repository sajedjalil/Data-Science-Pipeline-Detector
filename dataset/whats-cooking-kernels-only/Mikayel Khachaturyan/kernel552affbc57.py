import json
import pandas as pd
import numpy as np
import nltk

from collections import defaultdict

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import svm

with open('../input/whats-cooking-kernels-only/train.json') as f:
     train_data = pd.DataFrame(json.load(f))
with open('../input/whats-cooking-kernels-only/test.json') as f:
     test_data = pd.DataFrame(json.load(f))
        
def process_ingredients(ingredient_list):
    
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    
    separator = ' '
    tokenized_list = word_tokenize(separator.join(ingredient_list).lower())
    
    final_words = []
    word_Lemmatized = WordNetLemmatizer()
    
    for word, tag in pos_tag(tokenized_list):
        if word not in stopwords.words('english') and word.isalpha():
            final_word = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            final_words.append(final_word)
            
    return separator.join(final_words)

train_data['ingredients'] = train_data['ingredients'].apply(process_ingredients)
test_data['ingredients'] = test_data['ingredients'].apply(process_ingredients)

train_x = train_data['ingredients']
train_y = train_data['cuisine']

test_x = test_data['ingredients']

label_encoder = LabelEncoder()
train_y_encoded = label_encoder.fit_transform(train_y)

tfidf_vect = TfidfVectorizer(max_features=6000)
tfidf_vect.fit(train_x)
train_x_tfidf = tfidf_vect.transform(train_x)

test_x_tfidf = tfidf_vect.transform(test_x)

svm = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
svm.fit(train_x_tfidf,train_y_encoded)

predictions_svm = svm.predict(test_x_tfidf)

test_data['cuisine'] = label_encoder.inverse_transform(predictions_svm)
test_data[['id', 'cuisine']].to_csv('submission.csv', index=False)