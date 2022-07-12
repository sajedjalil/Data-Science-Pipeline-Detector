# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb # XGBoost implementation

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# read data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

features = [x for x in train.columns if x not in ['id','loss']]
#print(features)

cat_features = [x for x in train.select_dtypes(include=['object']).columns if x not in ['id','loss']]
num_features = [x for x in train.select_dtypes(exclude=['object']).columns if x not in ['id','loss']]
print(cat_features)
print(num_features)

train['log_loss'] = np.log(train['loss'])

train_x = train[features]
test_x = test[features]
for c in range(len(cat_features)):
    a = pd.DataFrame(train['log_loss'].groupby([train[cat_features[c]]]).mean())
    a[cat_features[c]] = a.index
    train_x[cat_features[c]] = pd.merge(left=train_x, right=a, how='left', on=cat_features[c])['log_loss']
    test_x[cat_features[c]] = pd.merge(left=test_x, right=a, how='left', on=cat_features[c])['log_loss']


xgdmat = xgb.DMatrix(train_x, train['log_loss']) # Create our DMatrix to make XGBoost more efficient

params = {'eta': 0.01, 'seed':0, 'subsample': 0.5, 'colsample_bytree': 0.5, 
             'objective': 'reg:linear', 'max_depth':6, 'min_child_weight':3} 

num_rounds = 1000
bst = xgb.train(params, xgdmat, num_boost_round = num_rounds)

test_xgb = xgb.DMatrix(test_x)
submission = pd.read_csv("../input/sample_submission.csv")
submission.iloc[:, 1] = np.exp(bst.predict(test_xgb))
submission.to_csv('xgb_starter.cat_mean.csv', index=None)