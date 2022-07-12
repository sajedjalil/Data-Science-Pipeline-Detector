import pandas as pd 

news = pd.read_csv('../input/nlp-getting-started/train.csv')

from sklearn.utils import shuffle

news = shuffle(news)
news = news.reset_index(drop = True)

news.drop(['keyword', 'location'], axis = 1, inplace = True)

news['text'] = news['text'].apply(lambda x: x.lower())

import string

def punctuation_removal(text):
	all_list = [char for char in text if char not in string.punctuation]
	clean_str = ''.join(all_list)
	return clean_str

news['text'] = news['text'].apply(punctuation_removal)

print (news)

import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
news['text'] = news['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(news['text'], news.target, test_size = 0.1, random_state = 42)

# Vectorizing and applying TF-IDF
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])

# Fitting the model
model = pipe.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

# Accuracy
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

test = pd.read_csv('../input/nlp-getting-started/test.csv')

test['text'] = test['text'].apply(lambda x: x.lower())

test['text'] = test['text'].apply(punctuation_removal)

prediction = model.predict(test['text'])
test['target'] = prediction
test.drop(['keyword', 'location', 'text'], axis = 1, inplace = True)

test.to_csv('./sub.csv', index = False, header = True)
