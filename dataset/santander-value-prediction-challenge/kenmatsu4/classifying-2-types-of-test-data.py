# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# This kernel is based on Prashant Kikani's  "santad_label is present in row?"
# https://www.kaggle.com/prashantkikani/santad-label-is-present-in-row

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import KFold
from sklearn import *

from collections import Counter

from datetime import  datetime as dt
def current_time():
    return dt.strftime(dt.now(),'%Y-%m-%d %H:%M:%S')

test = pd.read_csv('../input/test.csv')
testdex  = test.index

print('ip files loaded...!!')
only_one = test.columns[test.nunique() == 1]
test.drop(only_one, axis = 1, inplace = True)
te_ID = test['ID']
test.drop(['ID'], axis = 1, inplace = True)


is_artificial = []
for i in range(test.shape[0]):
    cnt_dict = dict(Counter((test.iloc[i]%1) ==0))
    if len(cnt_dict.keys()) != 2 or cnt_dict[False] != test.iloc[i].nunique()-1: 
        is_artificial.append(0)
    else:
        is_artificial.append(1)
        
print(len(is_artificial))

# for labeling
lgb_params = {
        'objective': 'binary',
        'num_leaves': 60,
        'subsample': 0.6143,
        'colsample_bytree': 0.6453,
        'min_split_gain': np.power(10, -2.5988),
        'reg_alpha': np.power(10, -2.2887),
        'reg_lambda': np.power(10, 1.7570),
        'min_child_weight': np.power(10, -0.1477),
        'verbose': -1,
        'seed': 3,
        'boosting_type': 'gbdt',
        'max_depth': -1,
        'learning_rate': 0.1,  # 0.05
        'metric': 'rmse',
    }

folds = KFold(n_splits=5, shuffle=True, random_state=1)

is_artificial = np.asarray(is_artificial)
n_not_artificial = len(is_artificial[is_artificial == 0])
n_artificial = len(is_artificial[is_artificial == 1])
print("n_not_artificial:", n_not_artificial) 
print("n_artificial:", n_artificial)
ratio = n_artificial/len(is_artificial)
print("ratio:", ratio)

print(test.shape, is_artificial.shape)

dtest = lgb.Dataset(data=test, label=is_artificial, free_raw_data=False)
oof_preds = np.zeros(test.shape[0])
for trn_idx, val_idx in folds.split(test):
        clf = lgb.train(
            params=lgb_params,
            train_set=dtest.subset(trn_idx),
            valid_sets=dtest.subset(val_idx),
            num_boost_round=10000,
            early_stopping_rounds=100,
            verbose_eval=500
        )
        oof_preds[val_idx] = clf.predict(dtest.data.iloc[val_idx])

preds = (oof_preds > ratio).astype(int)
print(is_artificial.shape, preds.shape)
print("accuracy_score:", accuracy_score(is_artificial, preds))
df_compare = pd.DataFrame([is_artificial, preds]).T
df_compare.columns = ['ground_truth', 'pred']
df_compare.to_csv('df_compare.csv', index = False)
