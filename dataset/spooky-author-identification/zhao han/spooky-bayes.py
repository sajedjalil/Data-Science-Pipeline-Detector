# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from nltk.stem.porter import PorterStemmer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string
import os

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, roc_auc_score, log_loss
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class model:
    def __init__(self):
        self.random_seed = 2000
        self.train_data = None
        self.test_data = None
        self.load_data()
        self.preprocess__data()
        self.feature_engineering()
        

    def load_data(self):
        self.train_data = pd.read_csv('../input/train.csv')
        self.test_data = pd.read_csv('../input/test.csv')

    """preprocess text:clean stopwords,clean non-numeric and non-word in text """

    def preprocess__data(self):
        combine = [self.train_data, self.test_data]
        ps = PorterStemmer()
        for data in combine:
            corpus = []
            for i in range(data.shape[0]):
                text = data['text'][i]
                review = re.sub('[^A-Za-z0-9]', ' ', text)
                review = word_tokenize(review)
                review = [word for word in str(review).lower().split() if word not in set(stopwords.words('english'))]
                review = [ps.stem(word) for word in review]
                review = ' '.join(review)
                corpus.append(review)
            data['clean'] = corpus
        print('preprocess_data ending ...')

    def get_corpus(self, text, ps, corpus):
        review = re.sub('[^A-Za-z0-9]', ' ', text)
        review = word_tokenize(review)
        review = [word for word in str(review).lower().split() if word not in stopwords.words('english')]
        review = [ps.stem(word) for word in review]
        review = ' '.join(review)
        corpus.append(review)
        

    def feature_engineering(self):
        combine = [self.train_data, self.test_data]
        
        for data in combine:
            data['num_words'] = data['text'].apply(lambda x: len(str(x).split()))
            data['num_uniq_words'] = data['text'].apply(lambda x: len(set(str(x).split())))
            data['num_word_stops'] = data['text'].apply(
                lambda x: len([word for word in str(x).lower().split() if word in stopwords.words('english')]))
            data['num_punctations'] = data['text'].apply(
                lambda x: len([word for word in str(x).split() if word in string.punctuation]))
            data['num_chars'] = data['text'].apply(lambda x: len(str(x)))
            data['word_upper'] = data['text'].apply(lambda x: len([word for word in str(x) if word is word.upper()]))
            data['word_title'] = data['text'].apply(lambda x: len([word for word in str(x) if word is word.title()]))
            data['word_mean'] = data['text'].apply(lambda x: np.mean([len(word) for word in str(x).lower().split()]))
            
        print('feature_engineering ending...')
        
        
    def train(self):
        tfidf = TfidfVectorizer(max_features=2000, dtype=np.float32, analyzer='word',
                        ngram_range=(1, 3), use_idf=True, smooth_idf=True,
                        sublinear_tf=True)
        x_train = tfidf.fit_transform(self.train_data['clean']).toarray()
        x_test = tfidf.fit_transform(self.test_data['clean']).toarray()
        name_ecode = {'EAP': 0, 'HPL': 1, 'MWS': 2}
        y = self.train_data['author'].map(name_ecode)
        kf = KFold(n_splits=10, shuffle=True, random_state=self.random_seed)
        mNB = MultinomialNB()
        predict_full_prob = 0
        predict_score = []
        count = 1
        for train_index, test_index in kf.split(x_train):
            print('{} of KFlod {}'.format(count, kf.n_splits))
            x1, x2 = x_train[train_index], x_train[test_index]
            y1, y2 = y[train_index], y[test_index]
            mNB.fit(x1, y1)
            y_predict = mNB.predict(x2)
            predict_score.append(log_loss(y2, mNB.predict_proba(x2)))
            predict_full_prob += mNB.predict_proba(x_test)
            count += 1
            print(predict_score)
        print('mean of predict score:{}'.format(np.mean(predict_score)))
        print('confusion matrix:\n', confusion_matrix(y2, y_predict))
        
        # y_pred = predict_full_prob / 10
        # submit = pd.DataFrame(self.test_data['id'])
        # submit = submit.join(pd.DataFrame(y_pred))
        # submit.columns = ['id', 'EAP', 'HPL', 'MWS']
        # # submit.to_csv('../input/spooky_pred1.csv.gz',index=False,compression='gzip')
        # submit.to_csv('../input/spooky_pred1.csv', index=False)
        


if __name__ == '__main__':
    m = model()
    m.train()

# Any results you write to the current directory are saved as output.