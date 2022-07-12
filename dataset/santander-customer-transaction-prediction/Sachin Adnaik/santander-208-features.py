# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn import metrics
import gc

pd.set_option('display.max_columns', 200)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# import data files
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

train_df.head()
test_df.head()

# Target variable distribution
train_df.target.value_counts()

# Adding features derived from row-wise summary of existing features

idx = features = train_df.columns.values[2:202]
for df in [train_df, test_df]:
    df['sum'] = df[idx].sum(axis=1)  
    df['min'] = df[idx].min(axis=1)
    df['max'] = df[idx].max(axis=1)
    df['mean'] = df[idx].mean(axis=1)
    df['std'] = df[idx].std(axis=1)
    df['skew'] = df[idx].skew(axis=1)
    df['kurt'] = df[idx].kurtosis(axis=1)
    df['med'] = df[idx].median(axis=1)

# Check updated data
train_df.head()

# LGBM parameters specification
param = {
    'num_leaves': 25,
     'max_bin': 60,
     'min_data_in_leaf': 5,
     'learning_rate': 0.0117,
     'min_sum_hessian_in_leaf': 0.0094,
     'feature_fraction': 0.057,
     'lambda_l1': 0.061,
     'lambda_l2': 4.659,
     'min_gain_to_split': 0.296,
     'max_depth': 50,
     'save_binary': True,
     'seed': 123,
     'feature_fraction_seed': 1234,
     'bagging_seed': 134,
     'drop_seed': 124,
     'data_random_seed': 234,
     'objective': 'binary',
     'boosting_type': 'gbdt',
     'verbose': 1,
     'metric': 'auc',
     'is_unbalance': True,
     'boost_from_average': False
}

# Folds
nfold = 10

# Specify target and predictors
target = 'target'
predictors = train_df.columns.values.tolist()[2:]

# Build model 
skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2019)

oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

i = 1
for train_index, valid_index in skf.split(train_df, train_df.target.values):
    print("\nfold {}".format(i))
    xg_train = lgb.Dataset(train_df.iloc[train_index][predictors].values,
                           label=train_df.iloc[train_index][target].values,
                           feature_name=predictors,
                           free_raw_data = False
                           )
    xg_valid = lgb.Dataset(train_df.iloc[valid_index][predictors].values,
                           label=train_df.iloc[valid_index][target].values,
                           feature_name=predictors,
                           free_raw_data = False
                           )   

    nround = 8000
    clf = lgb.train(param, xg_train, nround, valid_sets = [xg_valid], verbose_eval=250)
    oof[valid_index] = clf.predict(train_df.iloc[valid_index][predictors].values, num_iteration=nround) 
    
    predictions += clf.predict(test_df[predictors], num_iteration=nround) / nfold
    i = i + 1

print("\n\nCV AUC: {:<0.4f}".format(metrics.roc_auc_score(train_df.target.values, oof)))



# Submission file
submission = pd.DataFrame({"ID_code": test_df.ID_code.values})
submission["target"] = predictions
submission[:10]

submission.to_csv("LGMB_210_featutes.csv", index=False)