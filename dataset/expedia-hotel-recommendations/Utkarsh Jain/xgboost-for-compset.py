# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
import sys

from sklearn import cross_validation
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Experimental script - does not produce prediction file yet


def map5eval(preds, dtrain):
    actual = dtrain.get_label()
    predicted = preds.argsort(axis=1)[:,-np.arange(5)]
    metric = 0.
    for i in range(5):
        metric += np.sum(actual==predicted[:,i])/(i+1)
    metric /= actual.shape[0]
    return 'MAP@5', -metric

trainloc = "../input/train.csv"
testloc = "../input/test.csv"
ssloc = "../input/sample_submission.csv"
train_cols = ['site_name', 'user_location_region', 'is_package', 'srch_adults_cnt', 'srch_children_cnt','srch_destination_id', 'hotel_market', 'hotel_country', 'hotel_cluster']

params = {}
params['objective'] = 'multi:softprob'
params['eval_metric'] = 'mlogloss'
params['num_class'] = 100

df_train = pd.DataFrame(columns=train_cols)
# train_chunk = pd.read_csv(trainloc, chunksize=100000)
train_chunk = pd.read_csv(trainloc, chunksize=100000)
i = 0
for chunk in train_chunk:
    df_train = pd.concat([df_train, chunk[chunk['is_booking'] == 1][train_cols]])
    i = i + 1
    if i % 10 == 0:
        print("Rows loaded: " + str(i / 10) + "mn")

print(df_train.head())

for column in df_train:
    df_train[column] = df_train[column].astype(str).astype(int)

# print(df_train.shape())
x_train = df_train.drop(['hotel_cluster'],axis=1)
y_train = df_train['hotel_cluster'].values

# Create train datamatrix
# d_train = xgb.DMatrix(x_train, label=y_train)

# clf = xgb.cv(params, d_train, num_boost_round=100000, early_stopping_rounds=50, verbose_eval=True, metrics='map@5')
# clf = xgb.cv(params, d_train, num_boost_round=100000, early_stopping_rounds=50, verbose_eval=True)

clf = xgb.XGBClassifier(objective = 'multi:softmax',max_depth = 5,n_estimators=300,learning_rate=0.01,nthread=4,subsample=0.7,colsample_bytree=0.7,min_child_weight = 3,silent=False)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(df_train, y_train, stratify=y_train, test_size=0.2)
clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric=map5eval, eval_set=[(X_train, y_train),(X_test, y_test)])
