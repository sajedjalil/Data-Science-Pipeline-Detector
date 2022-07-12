__author__ = 'Andre Lopes'

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np
import xgboost as xgb
import pandas as pd

train = pd.read_csv('../input/train.csv')
train = train.drop('ID', axis=1)

y = train['TARGET']

train = train.drop('TARGET', axis=1)
x = train

dtrain = xgb.DMatrix(x.as_matrix(), label=y.tolist())

test = pd.read_csv('../input/test.csv')

test = test.drop('ID', axis=1)
dtest = xgb.DMatrix(test.as_matrix())


# XGBoost params:
def get_params():
    #
    params = {}
    params["objective"] = "binary:logistic"
    params["booster"] = "gbtree"
    params["eval_metric"] = "auc"
    params["eta"] = 0.3  #
    params["subsample"] = 0.50
    params["colsample_bytree"] = 1.0
    params["max_depth"] = 40
    params["nthread"] = 4
    plst = list(params.items())
    #
    return plst


bst = xgb.train(get_params(), dtrain, 10)

preds = bst.predict(dtest)

print ("preds.shape ", preds.shape)
print ("np.max preds = ",np.max(preds))
print ("np.min preds = ",np.min(preds))
print ("np.avg preds = ",np.average(preds))


for x in range(0,preds.shape[0]):
    if preds[x] > 0.500 :
        preds[x] = 1
    else:
        preds[x] = 0

print ("preds.shape ", preds.shape)
print ("np.max preds = ",np.max(preds))
print ("np.min preds = ",np.min(preds))
print ("np.avg preds = ",np.average(preds))

# Make Submission
test_aux = pd.read_csv('../input/test.csv')
result = pd.DataFrame({"Id": test_aux["ID"], 'TARGET': preds})

result.to_csv("xgboost_submission.csv", index=False)


