# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import csv
import re
import os
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE

import pandas as pd


from sklearn import tree
stemmer = SnowballStemmer("english")

def getwords(doc):
    splitter = re.compile('\\W+[0-9]*') # split with non-words
    stopworddic = set(stopwords.words('english'))
    words=[s.lower() for s in splitter.split(doc)
           if len(s)>2 and len(s)<20 and s not in stopworddic]
    stems = [stemmer.stem(t) for t in words]
    return stems
class Decision_Tree():
    def __init__(self):
        self.vec = []
        self.words = []
        self.result = []
        self.words_tst = []
        self.words_tr = []
        self.train_data = None
        self.test_data = None
        self.train_target = None
        self.key = []
    def read_csv(self):
        path_full = '../input\\train.csv'
        print('Load data...')
        path_test = '../input\\test.csv'
        self.words_tr = pd.read_csv(path_full)['comment_text'].values
        self.words_tst = pd.read_csv(path_test)['comment_text'].values
        self.key =  pd.read_csv(path_test)['id'].values

    def tf_idf(self):
        files = os.listdir('./')
        train_file = 'train_matrix1.pkl'
        test_file = 'test_matrix2.pkl'
        if (train_file not in files) or (test_file not in files):
            tfidf_vectorizer = TfidfVectorizer(max_df=0.5, max_features=250000,min_df=0.01,tokenizer=getwords, stop_words='english',
                                             use_idf=False, ngram_range=(1,1))
            tfidf_vectorizer.fit(self.words_tr)
            train_matrix = tfidf_vectorizer.transform(self.words_tr) #fit the vectorizer to synopses
            test_matrix = tfidf_vectorizer.transform(self.words_tst)
            joblib.dump(train_matrix, train_file)
            joblib.dump(test_matrix,test_file)
        else:
            train_matrix = joblib.load(train_file)
            test_matrix = joblib.load(test_file)
        self.vec = train_matrix
        self.test_data = test_matrix

    def balance_smote(self,target):
        sm = SMOTE(random_state=42)
        self.train_data, self.train_target = sm.fit_sample(self.vec, target)
        print ('SMOTE:')


    def save_prediction(self,label):
        print('Saving...')
        output = 'output1.csv'
        df = pd.DataFrame({label: self.result})
        if not os.path.isfile(output):
            df.to_csv(output)
        else:
            file = pd.read_csv(output)
            file[label] = self.result
            file.to_csv(output)
    def save_id(self):
        output = 'output1.csv'
        dataframe = pd.DataFrame({'id': self.key})
        dataframe.to_csv(output, index=False)
    def Decision_Tree(self):
        clf = tree.DecisionTreeClassifier()
        clf.fit(self.train_data, self.train_target)
        self.result = clf.predict(self.test_data)

if __name__ == '__main__':
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    Test = Decision_Tree()
    Test.read_csv()
    Test.tf_idf()
    Test.save_id()
    path_full = '../input\\train.csv'
    for i in labels:
        target_train = pd.read_csv(path_full)[i].values
        Test.balance_smote(target=target_train)
        Test.Decision_Tree()
        Test.save_prediction(label=i)