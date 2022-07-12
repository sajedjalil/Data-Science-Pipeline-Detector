# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 23:05:47 2016

@author: darshit
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
import gzip
import zipfile

number = preprocessing.LabelEncoder()
train = pd.read_csv('../input/train.csv', parse_dates=['Dates'], index_col=False)
test = pd.read_csv('../input/test.csv', parse_dates=['Dates'], index_col=False)

train = train.drop(['Descript', 'Resolution', 'Address'], axis = 1)
test = test.drop(['Address'], axis = 1)


def convert(data):
    number = preprocessing.LabelEncoder()
    data['Day'] = data['Dates'].dt.day
    data['Month'] = data['Dates'].dt.month
    data['Year'] = data['Dates'].dt.year
    data['Hour'] = data['Dates'].dt.hour
    data['Minute'] = data['Dates'].dt.minute
    data['DayOfWeek'] = data['Dates'].dt.dayofweek
    data['WeekOfYear'] = data['Dates'].dt.weekofyear
    data=data.fillna(0)
    return data

train=convert(train)
test = convert(test)

enc = LabelEncoder()
train['PdDistrict'] = enc.fit_transform(train['PdDistrict'])
category_encoder = LabelEncoder()
category_encoder.fit(train['Category'])
train['CategoryEncoded'] = category_encoder.transform(train['Category'])
enc = LabelEncoder()
test['PdDistrict'] = enc.fit_transform(test['PdDistrict'])
#print(train.columns)
#print(test.columns)

x_cols = list(train.columns[2:12].values)
x_cols.remove('Minute')
#print(x_cols)

#train, validation = train_test_split(train, test_size = 0.2)
clf = ensemble.RandomForestClassifier()
clf.set_params(n_estimators=40)
clf.set_params(min_samples_split=100)
clf.fit(train[x_cols], train['CategoryEncoded'])

test['predictions'] = clf.predict(test[x_cols])
pred = clf.predict_proba(test[x_cols])
#valid_pred = clf.predict_proba(validation[x_cols])
#print(valid_pred)
#score = metrics.log_loss(validation['CategoryEncoded'],valid_pred)
#print(score)

def field_to_columns(data, field, new_columns):
    for i in range(len(new_columns)):
        data[new_columns[i]] = (data[field] == new_columns[i]).astype(int)
    return data
    
'''test['Category'] = category_encoder.inverse_transform(test['predictions'])
categories = list(category_encoder.classes_)
test = field_to_columns(test, 'Category', categories)'''

submission = pd.DataFrame(pred,columns=category_encoder.classes_)
submission['Id'] = test.Id.tolist()

#submission_cols = [test.columns[0]]+list(test.columns[14:])
submission.to_csv(gzip.open('first_run.csv.gz','wt'), index = False)