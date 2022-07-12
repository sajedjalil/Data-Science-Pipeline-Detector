# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import GradientBoostingClassifier

print("loading")
train_df = pd.read_csv("../input/train.csv", index_col=None)
test_df = pd.read_csv("../input/test.csv", index_col=None)
train_df = train_df.replace(-999999,2)
print(train_df.shape)
print(test_df.shape)
# drop duplicates
def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []
    for t, v in groups.items():
        dcols = frame[v].to_dict(orient="list")

        vs = list(dcols.values())
        ks = list(dcols.keys())
        lvs = len(vs)
        for i in range(lvs):
            for j in range(i+1,lvs):
                if vs[i] == vs[j]: 
                    dups.append(ks[i])
                    break

    return dups

dup_list = duplicate_columns(train_df)
print(len(dup_list))

train_df = train_df.drop(dup_list, axis=1)
print(train_df.shape)
test_df =  test_df.drop(dup_list, axis=1)
print(test_df.shape)

X = train_df.iloc[:, 1:-1].values
y = train_df['TARGET'].values
X_test = test_df.iloc[:, 1:].values
print("X shape", X.shape)
print("X_test shape", X_test.shape)


# remove constant features
selector = VarianceThreshold(threshold = 0.001)
X = selector.fit_transform(X)
X_test = selector.transform(X_test)
print("After removing low variance features")
print("X shape:", X.shape)
print("X_test shape:", X_test.shape)


import xgboost as xgb
from sklearn.ensemble import BaggingClassifier

dtrain = xgb.DMatrix(X, label=y)
dtest = xgb.DMatrix(X_test)

evallist  = [(dtrain,'train')]

ypred_list = []

for seed in [1234]:
    param = {'max_depth':5, 
             'eta':0.02, 
             'silent':1, 
             'objective':'binary:logistic',
             'eval_metric': "auc",
             'subsample': 0.7,
             'colsample_bytree': 0.7,
             'booster': "gbtree",
             'seed': seed
             }
    
    num_round = 559
    plst = param.items()
    bst = xgb.train( plst, dtrain, num_round, evallist )
    
    ypred_list.append( bst.predict(dtest))

pred = np.mean(np.array(ypred_list), axis = 0)

submission = pd.DataFrame({"ID":test_df['ID'].values, "TARGET":pred})
submission.to_csv("submission.csv", index=False)



