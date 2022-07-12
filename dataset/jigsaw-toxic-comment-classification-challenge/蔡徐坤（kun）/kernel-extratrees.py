#!/usr/bin/env python3
# -*- coding: utf-8 -*-

########################################################################
## 导入模块
########################################################################

import os
os.chdir("/Users/lizhengfudan/Desktop/Kaggle_Toxic")

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, log_loss, auc
from textblob import TextBlob
from sklearn.model_selection import KFold

import re, string

########################################################################
## 导入文本数据与文本数据的简单处理
########################################################################

train = pd.read_csv('.../input/train.csv').fillna(' ')
test = pd.read_csv('.../input/test.csv').fillna(' ')
submission = pd.read_csv('.../input/sample_submission.csv')
label_cols = ['toxic', 'severe_toxic', 'obscene', \
              'threat', 'insult', 'identity_hate']

train['comment_text'].fillna("unknown", inplace=True)
test['comment_text'].fillna("unknown", inplace=True)

zpolarity = {0:'zero',1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine',10:'ten'}
zsign = {-1:'negative',  0.: 'neutral', 1:'positive'}

train['polarity'] = train['comment_text'].map(lambda x: int(TextBlob(x).sentiment.polarity * 10))
test['polarity'] = test['comment_text'].map(lambda x: int(TextBlob(x).sentiment.polarity * 10))

train['comment_text'] = train.apply(lambda r: str(r['comment_text']) + ' polarity' +  zsign[np.sign(r['polarity'])] + zpolarity[np.abs(r['polarity'])], axis=1)
test['comment_text'] = test.apply(lambda r: str(r['comment_text']) + ' polarity' +  zsign[np.sign(r['polarity'])] + zpolarity[np.abs(r['polarity'])], axis=1)

Features_train_polarity = train['polarity']
Features_train_Name = 'Features_train_polarity'
Features_train_polarity.to_csv(Features_train_Name)

Features_test_polarity = test['polarity']
Features_test_Name = 'Features_test_polarity'
Features_test_polarity.to_csv(Features_test_Name)

########################################################################
## 构造 TFIDF 特征
########################################################################

vectorizer = TfidfVectorizer(stop_words='english', 
                             ngram_range=(1, 2), 
                             max_features=800000)

all_text = pd.concat([train['comment_text'], test['comment_text']])
vectorizer.fit(all_text)
train_features = vectorizer.transform(train['comment_text'])
test_features = vectorizer.transform(test['comment_text'])
y_train_array = train[label_cols].values

########################################################################
## 训练 extratrees 模型
########################################################################

folds = KFold(n_splits = 5, shuffle = True, random_state = 1)
Len_train = train.shape[0]
Len_test = test.shape[0]
Len = int( np.ceil(Len_train / 5) )
perm = np.random.permutation(Len_train)
y_cv_pred = np.zeros((Len_train, 6))
y_test_pred = np.zeros((Len_test, 6, 5))
y_train_selected_pred = np.zeros((Len_train, 6, 5))
for k, (idx1, idx2) in enumerate(folds.split(train_features, y_train_array)):
    
    ## 建立训练集与交叉检验集    
    X_train_selected = train_features[idx1, :]
    y_train_selected = y_train_array[idx1, :]
    X_cv = train_features[idx2, :]
    y_cv = y_train_array[idx2, :]
    
    ## 针对每个特征做计算
    model = ExtraTreesClassifier(n_jobs=4, random_state=1, verbose = 1)
    model.fit(X_train_selected, y_train_selected)
    
    data1 = model.predict_proba(X_cv)
    data2 = model.predict_proba(test_features)
    data3 = model.predict_proba(X_train_selected)
        
    for n in range(0, 5):
        y_cv_pred[idx2, n] = data1[n][:,1]
        y_test_pred[:, n, k] = data2[n][:,1]
        y_train_selected_pred[idx1, n, k] = data3[n][:,1]
        
    ## 计算auc并保存
    test_auc_logloss = np.zeros((6, 4))
    for i in range(0, 6):
        test_auc_logloss[i, 0] = roc_auc_score(y_cv[:, i], y_cv_pred[idx2, i])
        test_auc_logloss[i, 1] = log_loss(y_cv[:, i], y_cv_pred[idx2, i])
        test_auc_logloss[i, 2] = roc_auc_score(y_train_selected[:, i], \
                        y_train_selected_pred[idx1, i, k])
        test_auc_logloss[i, 3] = log_loss(y_train_selected[:, i], \
                        y_train_selected_pred[idx1, i, k])
    print(test_auc_logloss.mean(axis = 0))
        
########################################################################
## 合并最终的预测结果
########################################################################

STMT = 'extratrees'
Submission_Name = 'Submission_' + STMT + '.csv'
submission[label_cols] = y_test_pred.mean(axis = 2)
submission.to_csv(Submission_Name, index=False)

CV_Name = 'CV_' + STMT + '.csv'
y_cv_pred = pd.DataFrame(y_cv_pred)
y_cv_pred.columns = label_cols
y_cv_pred.to_csv(CV_Name)

cv_auc_logloss = np.zeros((6, 2))
y_cv_pred = np.array(y_cv_pred)
for i in range(0, 6):
    cv_auc_logloss[i, 0] = roc_auc_score(y_train_array[:, i], y_cv_pred[:, i])
    cv_auc_logloss[i, 1] = log_loss(y_train_array[:, i], y_cv_pred[:, i])
print(cv_auc_logloss.mean(axis = 0))

