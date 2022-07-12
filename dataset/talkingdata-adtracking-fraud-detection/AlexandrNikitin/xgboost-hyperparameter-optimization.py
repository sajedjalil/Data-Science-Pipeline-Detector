# uses "hyperopt" library, more info https://github.com/hyperopt/hyperopt
# XGBoost code based on https://www.kaggle.com/alexandrnikitin/efficient-xgboost-on-sparse-matrices

import gc
import os
import operator

from glob import glob

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from pandas.core.categorical import Categorical
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import roc_auc_score

dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8'
}
to_read = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
to_parse = ['click_time']

# Features used in training
categorical_features = ['app', 'device', 'os', 'channel']
numerical_features = ['clicks_by_ip']

train_size = 8000000

def sparse_dummies(df, column):
    """Returns sparse OHE matrix for the column of the dataframe"""
    categories = Categorical(df[column])
    column_names = np.array([f"{column}_{str(i)}" for i in range(len(categories.categories))])
    N = len(categories)
    row_numbers = np.arange(N, dtype=np.int)
    ones = np.ones((N,))
    return csr_matrix((ones, (row_numbers, categories.codes))), column_names

df_train = pd.read_csv('../input/train.csv', nrows=10000000, usecols=to_read, dtype=dtypes, parse_dates=to_parse)

# Example of numerical feature
# warning: this is for the sake of example; the feature leaks: it doesn't take time into account, takes whole train dataset, doesn't cut off time; 
clicks_by_ip = df_train.groupby(['ip']).size().rename('clicks_by_ip', inplace=True)
df_train = df_train.join(clicks_by_ip, on='ip')
del clicks_by_ip
gc.collect()

matrices = []
all_column_names = []
# creates a matrix per categorical feature
for c in categorical_features:
    matrix, column_names = sparse_dummies(df_train, c)
    matrices.append(matrix)
    all_column_names.append(column_names)

# appends a matrix for numerical features (one column per feature)
matrices.append(csr_matrix(df_train[numerical_features].values, dtype=float))
all_column_names.append(df_train[numerical_features].columns.values)

train_sparse = hstack(matrices, format="csr")
feature_names = np.concatenate(all_column_names)
del matrices, all_column_names

X = train_sparse
y = df_train['is_attributed']

del df_train
gc.collect()

# Create binary training and validation files for XGBoost
x1, y1 = X[:train_size], y.iloc[:train_size]
dm1 = xgb.DMatrix(x1, y1, feature_names=feature_names)
dm1.save_binary('train.bin')
del dm1, x1, y1
gc.collect()

x2, y2 = X[train_size:], y.iloc[train_size:]
dm2 = xgb.DMatrix(x2, y2, feature_names=feature_names)
dm2.save_binary('validate.bin')
del dm2, x2, y2
del X, y, train_sparse
gc.collect()

# XGBoost parameters example
params = {
    'eta': 0.3,
    'tree_method': "hist",
    'grow_policy': "lossguide",
    'max_leaves': 1000,  
    'max_depth': 0, 
    'subsample': 0.9, 
    'alpha':1,
    'objective': 'binary:logistic', 
    'scale_pos_weight':100,
    'eval_metric': 'auc', 
    'nthread':4,
    'silent': 1
}

# Pointers to binary files for training and validation
# They won't be loaded into Python environment but passed directly to XGBoost
dmtrain = xgb.DMatrix('train.bin', feature_names=feature_names)
dmvalid = xgb.DMatrix('validate.bin', feature_names=feature_names)



# Hyperparameter optimization:

# objective function to optimize; loss is auroc
def objective(params):
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    watchlist = [(dmtrain, 'train'), (dmvalid, 'valid')]
    model = xgb.train(params, dmtrain, num_round, watchlist, maximize=True, early_stopping_rounds=20, verbose_eval=1)
    pred = model.predict(dmvalid, ntree_limit=model.best_ntree_limit)
    auc = roc_auc_score(dmvalid.get_label(), pred)
    del model, pred
    gc.collect()
    print(f"SCORE: {auc}")
    return { 'loss': 1-auc, 'status': STATUS_OK }

# hyperparameter optimization space
# find more parameters in docs https://github.com/dmlc/xgboost/blob/443ff746e9723dcf571769b0d6ea28fbcb3e4a3f/doc/parameter.md
space = {
    # 'n_estimators': hp.quniform('n_estimators', 200, 600, 50),
    'n_estimators': 3, # WARNING: increse number of estimators, e.g. uncomment the above line (it's small for the sake of example)
    'eta': hp.quniform('eta', 0.025, 0.25, 0.025),
    'max_depth': hp.choice('max_depth', np.arange(1, 14, dtype=int)),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'subsample': hp.quniform('subsample', 0.7, 1, 0.05),
    'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 1, 0.05),
    'alpha' : hp.quniform('alpha', 0, 10, 1),
    'lambda': hp.quniform('lambda', 1, 2, 0.1),
    'scale_pos_weight': hp.quniform('scale_pos_weight', 50, 200, 10),
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': "hist",
    'booster': 'gbtree',
    'nthread': 4, 
    'silent': 1
}

trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=10, # WARNING: increase number of evaluations (it's small for the sake of example)
    trials=trials
)

# best hyperparameters
print("\n\n\n The best hyperparameters:")
print(best)
