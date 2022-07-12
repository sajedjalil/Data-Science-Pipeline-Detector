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

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# --- Load Training and Test Data ---
train = pd.read_csv('../input/train.csv')#[:1000]
test  = pd.read_csv('../input/test.csv')#[:1000]
sample = pd.read_csv('../input/sample_submission.csv')#[:1000]    



# --- Data Prep ---
train['comment_text'] = train['comment_text'].fillna('__nocomment__')
test['comment_text']  = test['comment_text'].fillna('__nocomment__')

train['comment_text'] = train['comment_text'].map(lambda x : x.lower())
test['comment_text'] = test['comment_text'].map(lambda x : x.lower())



# --- Custom Utility for AUC ---
def customAuc(yActual, yPred):
    fpr, tpr, __ = metrics.roc_curve(yActual, yPred)
    auc          = metrics.auc(fpr, tpr)
    return auc



# --- Split Training Data : Train and Validation ---
columnList = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
xTrain, xValid, yTrain, yValid = train_test_split(train.comment_text.values, train[columnList],
                                                  random_state=42, test_size=0.25, shuffle=True)



# --- TFIDF & WordCount ---
tfv = TfidfVectorizer(min_df=3,  max_features=10000, strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                      ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1, stop_words = 'english')
cfv = TfidfVectorizer(min_df=3,  max_features=10000, strip_accents='unicode', analyzer='char',token_pattern=r'\w{1,}',
                      ngram_range=(2, 6), use_idf=1,smooth_idf=1,sublinear_tf=1, stop_words = 'english')



# --- Features ---
train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])
del train, test


tfv.fit(all_text)
cfv.fit(all_text)

xTrainTfv = tfv.transform(xTrain)
xValidTfv = tfv.transform(xValid)
xTestTfv  = tfv.transform(test_text.values)

xTrainCfv = cfv.transform(xTrain)
xValidCfv = cfv.transform(xValid)
xTestCfv  = cfv.transform(test_text.values)
del xTrain, xValid

xTrainStack = hstack([xTrainTfv, xTrainCfv])
xValidStack = hstack([xValidTfv, xValidCfv])
xTestStack  = hstack([xTestTfv, xTestCfv])


del xTrainTfv, xValidTfv, xTestTfv
del xTrainCfv, xValidCfv, xTestCfv



# --- RF Classifier ---
mlpClass =MLPClassifier(solver='lbfgs', alpha=1e-5, validation_fraction=0.3, hidden_layer_sizes=(4,4), verbose= True, activation= 'logistic', max_iter= 200 , learning_rate_init= 0.0001)
mlpClass.fit(xTrainStack, yTrain)



predValid           = pd.DataFrame(mlpClass.predict_proba(xValidStack))
predValid.columns   = columnList
print(predValid.head())
predTest            = pd.DataFrame(mlpClass.predict_proba(xTestStack))
predTest.columns    = columnList
predTest.ix[:, 'id'] = sample['id']



print(mlpClass.score(xValidStack, yValid))



logLossValid = []
for col in columnList:
    ll = customAuc(yValid[col], predValid[col])
    logLossValid.append(ll)



print('MLPC AUC :', np.mean(logLossValid))



# --- Output ---
predTest.to_csv('MLPClassifier.csv', index = False)


