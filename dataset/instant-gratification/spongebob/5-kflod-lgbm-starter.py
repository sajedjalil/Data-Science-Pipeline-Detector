#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd 
import os
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from tqdm import tqdm
from sklearn.preprocessing import scale, minmax_scale
from scipy.stats import norm

random_state = 42
np.random.seed(random_state)
df_train = pd.read_csv('../input/train.csv')[:]
df_test = pd.read_csv('../input/test.csv')[:]


train = df_train.copy()
test = df_test.copy()
features_original = [c for c in train.columns if c not in ['id', 'target']]

len_train = len(train)
train = train.append(test).reset_index(drop = True)



test = train[len_train:]
train = train[:len_train]
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'verbose': 1,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.7,
    'min_data_in_leaf': 200,
    'bagging_fraction': 0.8,
    'bagging_freq': 20,
    'min_hessian': 0.01,
    'feature_fraction_seed': 2,
    'bagging_seed': 3,
    "seed": random_state
}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
oof = train[['id', 'target']]
oof['predict'] = 0
predictions = test[['id']]
val_aucs = []

features = [col for col in test.columns if col not in ['target', 'id']]
X_test = test[features].values

for fold, (trn_idx, val_idx) in enumerate(skf.split(train, train['target'])):
    X_train, y_train = train.iloc[trn_idx][features], train.iloc[trn_idx]['target']
    X_valid, y_valid = train.iloc[val_idx][features], train.iloc[val_idx]['target']
    trn_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    evals_result = {}
    lgb_clf = lgb.train(lgb_params,
                        trn_data,
                        7500,
                        valid_sets=[val_data],
                        early_stopping_rounds=100,
                        verbose_eval=50,
                        evals_result=evals_result)

    p_valid = lgb_clf.predict(X_valid[features], num_iteration=lgb_clf.best_iteration)

    oof['predict'][val_idx] = p_valid
    val_score = roc_auc_score(y_valid, p_valid)
    val_aucs.append(val_score)

    predictions['fold{}'.format(fold + 1)] = lgb_clf.predict(test[features],
                   num_iteration=lgb_clf.best_iteration)

mean_auc = np.mean(val_aucs)
std_auc = np.std(val_aucs)
all_auc = roc_auc_score(oof['target'], oof['predict'])
print("Mean auc: %.9f, std: %.9f. All auc: %.9f." % (mean_auc, std_auc, all_auc))

predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['id', 'target']]].values, axis=1)
sub_df = pd.DataFrame({"id":df_test["id"].values})
sub_df["target"] = predictions['target']
sub_df.to_csv("bao_lgb_sub_{}.csv".format(all_auc), index=False)
oof.to_csv('bao_lgb_oof_{}.csv'.format(all_auc), index=False)
predictions.to_csv('bao_lgb_all_predictions_{}.csv'.format(all_auc), index=False)
predictions[['id', 'target']].to_csv('submission.csv', index=False)