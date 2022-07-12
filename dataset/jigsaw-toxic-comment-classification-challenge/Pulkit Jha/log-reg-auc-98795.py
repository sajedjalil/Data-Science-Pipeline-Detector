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


import warnings
warnings.simplefilter("ignore", UserWarning)


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


train = pd.read_csv('../input/train.csv')[:10000]
test = pd.read_csv('../input/test.csv')[:10000]
sample = pd.read_csv('../input/sample_submission.csv')[:10000]


train['comment_text'] = train['comment_text'].fillna('__nocomment__')
test['comment_text']  = test['comment_text'].fillna('__nocomment__')

train['comment_text'] = train['comment_text'].map(lambda x : x.lower())
test['comment_text'] = test['comment_text'].map(lambda x : x.lower())


def customAuc(yActual, yPred):
    fpr, tpr, __ = metrics.roc_curve(yActual, yPred)
    auc          = metrics.auc(fpr, tpr)
    return auc


columnList = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

xTrain, xValid, yTrain, yValid = train_test_split(train.comment_text.values, train[columnList],
                                                  random_state=42, test_size=0.1, shuffle=True)


#Building Basic Models
#TFIDF Text Vectorizer
tfv = TfidfVectorizer(min_df=3,  max_features=20000, strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                      ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1, stop_words = 'english')
cfv = TfidfVectorizer(min_df=3,  max_features=50000, strip_accents='unicode', analyzer='char',token_pattern=r'\w{1,}',
                      ngram_range=(2, 6), use_idf=1,smooth_idf=1,sublinear_tf=1, stop_words = 'english')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

#tfv.fit(list(xTrain) + list(xValid))
tfv.fit(all_text)
cfv.fit(all_text)

xTrainTfv = tfv.transform(xTrain)
xValidTfv = tfv.transform(xValid)
xTestTfv  = tfv.transform(test.comment_text.values)

xTrainCfv = cfv.transform(xTrain)
xValidCfv = cfv.transform(xValid)
xTestCfv  = cfv.transform(test.comment_text.values)

xTrainStack = hstack([xTrainTfv, xTrainCfv])
xValidStack = hstack([xValidTfv, xValidCfv])
xTestStack  = hstack([xTestTfv, xTestCfv])


# Fitting a simple Logistic Regression on TFIDF
# --- Parameter Tuning ---
# --- Added Class Weight : Score Improved ---
# --- Changed Penalty to l1 : No improvement ---
# --- CV : No improvement ---

predValid = pd.DataFrame()
predTest = pd.DataFrame()
predTest.ix[:, 'id'] = sample['id']
clf = LogisticRegression(C= 1, class_weight='balanced')
#cList = [0.001, 0.01, 0.1, 1, 10, 100] 
#clf = LogisticRegressionCV(Cs=cList, class_weight='balanced')
#model = GridSearchCV(estimator=clf, param_grid=param_grid,
#                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)
for col in columnList:
    #clf.fit(xTrainTfv, yTrain[col])
    clf.fit(xTrainStack, yTrain[col])
    print('Fitting ...', col)
    #print('Optimal Value of C ...:', clf.C_)
    predValid.ix[:, col] = clf.predict_proba(xValidStack)[:,1]
    predTest.ix[:, col] = clf.predict_proba(xTestStack)[:,1]


logLossValid = []
for col in columnList:
    ll = customAuc(yValid[col], predValid[col])
    logLossValid.append(ll)


print('Logistic AUC :', np.mean(logLossValid))

