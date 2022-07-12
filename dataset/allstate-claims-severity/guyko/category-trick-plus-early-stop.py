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
#print(cat_features)
#print(num_features)

train['log_loss'] = np.log(train['loss'])

ntrain = train.shape[0]
ntest = test.shape[0]
train_test = pd.concat((train[features], test[features])).reset_index(drop=True)
cat_features_2 = []
cat_features_3_10 = []
cat_features_10 = []
for c in range(len(cat_features)):
    if len(train_test[cat_features[c]].unique()) == 2:
        cat_features_2.append(cat_features[c])
    if 3 <= len(train_test[cat_features[c]].unique()) <= 10:
        cat_features_3_10.append(cat_features[c])
    if len(train_test[cat_features[c]].unique()) > 10:
        cat_features_10.append(cat_features[c])

#get dummy on categorical variables where unique values are between 3 and 10
train_test = pd.concat([pd.get_dummies(train_test[cat_features_3_10]), train_test], axis = 1)
for c in range(len(cat_features_3_10)):
    del train_test[cat_features_3_10[c]]

#get binary on categorical variables where unique values are 2
for c in range(len(cat_features_2)):
    train_test[cat_features_2[c]] = train_test[cat_features_2[c]].astype('category').cat.codes

#get a random 10-dim encoder on categorical variables where unique values are greater than 10
for c in range(len(cat_features_10)):
    l = len(train_test[cat_features_10[c]].unique())
    mmult = np.random.rand(l,10)
    a = pd.DataFrame(np.dot(pd.get_dummies(train_test[cat_features_10[c]]),mmult))
    cols = []
    for l in range(10):
        cols.append(cat_features_10[c] + "_" + str(l))
    a.columns = cols
    train_test = pd.concat([a, train_test], axis = 1)
    del train_test[cat_features_10[c]]
    
     

train_x = train_test.iloc[:ntrain,:]
test_x = train_test.iloc[ntrain:,:]

from datetime import datetime

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(' Time taken: %i minutes and %s seconds.' %
              (tmin, round(tsec, 2)))
              
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
# enter the number of folds from xgb.cv
folds = 5
cv_sum = 0
early_stopping = 25
fpred = []
xgb_rounds = []

start_time = timer(None)

# XGBoost data structure
d_train_full = xgb.DMatrix(train_x, label=train['log_loss'])
d_test = xgb.DMatrix(test_x)



####################################
#  Make Full Dataset Predictions
####################################
params = {'eta': 0.05, 'seed':0, 'subsample': 0.5, 'colsample_bytree': 0.5, 
             'objective': 'reg:linear', 'max_depth':6, 'min_child_weight':3} 

# Grid Search CV optimized settings
num_rounds = 200
bst = xgb.train(params, d_train_full, num_boost_round = num_rounds)

submission = pd.read_csv("../input/sample_submission.csv")
submission.iloc[:, 1] = np.exp(bst.predict(d_test))
submission.to_csv('xgb_starter.sub.csv', index=None)