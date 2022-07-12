

# --- Import Modules ---
import numpy as np
import pandas as pd
import seaborn as sns
import re
import os 
from datetime import datetime 


from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 
from sklearn.model_selection import train_test_split


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# --- Load Training and Test Data ---
train = pd.read_csv('../input/train.csv')[:10000]
test  = pd.read_csv('../input/test.csv')[:10000]
sample = pd.read_csv('../input/sample_submission.csv')[:10000]    



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
predValid = pd.DataFrame()
predTest = pd.DataFrame()
predTest.ix[:, 'id'] = sample['id']
clf = RandomForestClassifier(n_estimators=100, max_features = 'log2', min_samples_leaf = 50, n_jobs = -1, random_state = 111, oob_score = True)

for col in columnList:
    print('Fitting ...', col)
    clf.fit(xTrainStack, yTrain[col])
    predValid.ix[:, col] = clf.predict_proba(xValidStack)[:,1]
    predTest.ix[:, col] = clf.predict_proba(xTestStack)[:,1]


logLossValid = []
for col in columnList:
    ll = customAuc(yValid[col], predValid[col])
    logLossValid.append(ll)
print('RandomForest AUC :', np.mean(logLossValid))



predTest.to_csv('RFClassifier.csv', index = False)