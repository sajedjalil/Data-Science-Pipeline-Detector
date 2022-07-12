# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')
data.head()

test = pd.read_csv('../input/test.csv')
test["TARGET"] = -1
test.head()

cols_to_drop = ['ID']
for i in data:
    if sum(data[i])==0:
        cols_to_drop.append(j)
    for j in data:
        if j not in cols_to_drop:
            if all(data[i]==data[j]) and i!=j:
                cols_to_drop.append(j)
        
test_colsDrop = cols_to_drop
cols_to_drop.append('TARGET')
print(cols_to_drop)

train = data.drop(cols_to_drop,axis=1)
label = data['TARGET']

def eval_wrapper(yhat, y):  
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = (yhat>0.5).astype(int)
    return accuracy_score(yhat, y)

def get_params():
    
    params = {}
    params["objective"] = "binary:logistic"
    params["booster"] = "gbtree"
    params["eta"] = 0.01
    params["eval_metric"] = "auc"
    params["subsample"] = 0.9
    params["colsample_bytree"] = 0.85
    params["max_depth"] = 6
    plst = list(params.items())

    return plst

# convert data to xgb data structure
xgtrain = xgb.DMatrix(train, label.values)

# get the parameters for xgboost
plst = get_params()
print(plst)

# global variables
xgb_num_rounds = 1000

# train model
model = xgb.train(plst, xgtrain, xgb_num_rounds, maximize=False, verbose_eval=True)

#get predictions
xgtest = xgb.DMatrix(test.drop(test_colsDrop,axis=1), test['TARGET'].values)
test_preds = model.predict(xgtest, ntree_limit=model.best_iteration)

#Submission
preds_out = pd.DataFrame({"ID": test['ID'].values, "TARGET": test_preds})
preds_out = preds_out.set_index('ID')
preds_out.to_csv('submission_5.csv')
