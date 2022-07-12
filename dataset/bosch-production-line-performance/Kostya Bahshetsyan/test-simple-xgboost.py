# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import log_loss, roc_auc_score
import xgboost as xgb
import time
import gc

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
datadir = '../input/'

tmp = time.time()
# Any results you write to the current directory are saved as output.
train_categorical = pd.read_csv(os.path.join(datadir,'train_categorical.csv'), nrows = 1000)                      
train_date = pd.read_csv(os.path.join(datadir,'train_date.csv'), nrows = 1000)
train_numeric = pd.read_csv(os.path.join(datadir,'train_numeric.csv'), nrows = 1000)

test_categorical = pd.read_csv(os.path.join(datadir,'test_categorical.csv'), nrows = 1000)                      
test_date = pd.read_csv(os.path.join(datadir,'test_date.csv'), nrows = 1000)
test_numeric = pd.read_csv(os.path.join(datadir,'test_numeric.csv'), nrows = 1000)

train = train_numeric.merge(train_date)
train = train.merge(train_categorical)
train.fillna(0, inplace=True)
train.info()

test = test_numeric.merge(test_date)
test = test.merge(test_categorical)
test.fillna(0, inplace=True)
test.info()
print('Load data in: %s seconds' % (str(time.time() - tmp)))

probs = {}
cat = []
print('Select feature:')
for x in train.columns.values:
    for a in train[x].unique():
        probs[a] = train.loc[train[x] == a]['Response'].mean()
    preds_train = [probs[a] for a in train[x]]
    if roc_auc_score(train['Response'], preds_train)>0.9:
    	cat.append(x)
print('Done select feature in: %s seconds' % (str(time.time() - tmp)))

y = train.Response
train=train.drop(['Id','Response'],axis=1)

Id = test.Id
test=test.drop('Id',axis=1)

chars = [i for i in cat if i in train.columns.values]

print('Ready for XGBClassifier: %s seconds' % (str(time.time() - tmp)))
gbm = xgb.XGBClassifier(silent=False, nthread=4, max_depth=10, n_estimators=800, subsample=0.5, learning_rate=0.03, seed=1337)
print('Done XGBClassifier: %s seconds' % (str(time.time() - tmp)))

gbm.fit(train[chars], y)
ypred = gbm.predict(test[chars])

output = pd.DataFrame({ 'Id' : Id, 'Response': ypred })
output.head()
output.to_csv('result.csv', index = False)





