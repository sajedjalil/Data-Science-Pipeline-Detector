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
import sys
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def preprocess_review(df):
    # Remove HTML tags and stopwords
    # df: pd.Series containing raw reviews
    result = []
    stw = stopwords.words('english')
    for i in tqdm(range(df.shape[0])):
        rev = df[i]
        rev_notag = BeautifulSoup(rev, 'lxml').get_text()
        rev_processed = [ word for word in re.sub('[^a-zA-Z]', ' ', rev_notag).lower().split(' ') if word not in stw + ['']]
        result.append(' '.join(rev_processed))
    return result

def to_Vocab(df):
    # Return CountVectorizer instance trained with df
    # df: pd.Series containing preprocessed reviews
    countvectorizer = CountVectorizer(dtype=np.uint8) # To avoid MemoryError
    countvectorizer.fit(df)
    return countvectorizer

def to_BoW(df, countvectorizer):
    # Convert pd.Series containing preprocessed reviews into
    # NumPy ndarray with countvectorizer
    return countvectorizer.transform(df).toarray()
    
def make_csv(df_id, df_sentiment, csvname):
    # df: pd.Series containing estimated labels for test data
    newdf = pd.DataFrame({'id':df_id, 'sentiment':df_sentiment})
    newdf.to_csv(csvname, index=False)
    return newdf

def size(obj):
    s = sys.getsizeof(obj)
    t = str(s)
    target = None
    for k, v in globals().items():
        if id(v) == id(obj):
            target = k
    if target:
        out = t[:(len(t)%3 if len(t)%3 else 3)] + (('.'+t[(len(t)%3 if len(t)%3 else 3):]+('',' KB',' MB',' GB')[(len(t)-1)//3]) if (len(t)-1)//3 else ' Bytes')
        print('size of \'{}\': {}'.format(target, out))
    else:
        print('Object not exist')
    
# Training Data
data_train = pd.read_table('../input/labeledTrainData.tsv')
# Test Data
data_sub = pd.read_table('../input/testData.tsv')
data_sub_id = data_sub['id']

print(data_train.head())
print(data_sub.head())


# Preprocess Raw Reviews
data_train_processed = preprocess_review(data_train['review'])
data_sub_processed = preprocess_review(data_sub['review'])

print(data_train_processed[:5])
print(data_sub_processed[:5])

size(data_train)
size(data_sub)
size(data_sub_id)
size(data_train_processed)
size(data_sub_processed)

# Convert into BoW
myvectorizer = to_Vocab(data_train_processed)
size(myvectorizer)
X_train = to_BoW(data_train_processed, myvectorizer)
size(X_train)
X_sub = to_BoW(data_sub_processed, myvectorizer)

print(X_train[:5])
print(X_sub[:5])

# Data Split
T_train = data_train['sentiment']
X_tr, X_va, T_tr, T_va = train_test_split(X_train, T_train, random_state=0)

print(len(X_tr), len(X_va), len(T_tr), len(T_va))

# Train and Predict
forest = RandomForestClassifier(n_estimators=100, random_state=0, verbose=True)
forest.fit(X_tr, T_tr)
Y_tr = forest.predict(X_tr)
Y_tr_p = forest.predict_proba(X_tr)
Y_va = forest.predict(X_va)
Y_va_p = forest.predict_proba(X_va)
print(Y_tr[:5])
print(Y_tr_p[:5])
print(Y_va[:5])
print(Y_va_p[:5])

# Evaluation
args_f_tr = (T_tr.values, Y_tr)
args_m_tr = (T_tr.values, Y_tr_p[:,1])
args_f_va = (T_va.values, Y_va)
args_m_va = (T_va.values, Y_va_p[:,1])

auc_tr, auc_va = roc_auc_score(*args_m_tr), roc_auc_score(*args_m_va)
print('AUC Score = {},{}'.format(auc_tr, auc_va))

# For Submission
Y_sub = forest.predict(X_sub)
print(Y_sub[:5])
print(make_csv(data_sub_id, Y_sub, 'submission.csv')[:5])