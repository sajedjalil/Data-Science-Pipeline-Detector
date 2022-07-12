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
df_train = pd.read_csv('../input/train.csv')[:200]
df_test = pd.read_csv('../input/test.csv')[:200]


train = df_train.copy()
test = df_test.copy()
features_original = [c for c in train.columns if c not in ['ID_code', 'target']]

test_values = test[features_original].values
unique_samples = []
unique_count = np.zeros_like(test_values)
for feature in tqdm(range(test_values.shape[1])):
    _, index_, count_ = np.unique(test_values[:, feature], return_counts=True, return_index=True)
    unique_count[index_[count_ == 1], feature] += 1

# Samples which have unique values are real the others are fake
real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]


real_test = test.iloc[real_samples_indexes]
#test.iloc[synthetic_samples_indexes]
total = pd.concat([train, real_test])
for f in tqdm(features_original):
    vc = total[f].value_counts()
    f_map = vc#dict(zip(vc.keys(), minmax_scale(vc.values)))
    train['{}_ID_code_counts'.format(f)] = train[f].map(f_map)
    test['{}_ID_code_counts'.format(f)] = test[f].map(f_map)


def get_hist_cum(total, f):
    v_c = total[f].value_counts()
    v_c = v_c.reset_index()
    v_c.sort_values(by = 'index', inplace = True, axis = 0)
    v_c['cum_count{}'.format(f)] = v_c[f].cumsum()
    return v_c[['index','cum_count{}'.format(f)]]

len_train = len(train)
train = train.append(test).reset_index(drop = True)


def get_hist_mean(total, f):
    vc = total[f].value_counts().reset_index()
    v_mean = total[f].mean()
    lt_mean_vc = vc.loc[vc['index']<v_mean]
    lt_mean_vc.sort_values(by=['index'],ascending=False,inplace=True)
    lt_mean_vc['{}_offset_mean_cum_count'.format(f)] =  lt_mean_vc[f].cumsum()
    ge_mean_vc = vc.loc[vc['index']>=v_mean]
    ge_mean_vc.sort_values(by=['index'],inplace=True)
    ge_mean_vc['{}_offset_mean_cum_count'.format(f)] = ge_mean_vc[f].cumsum()
    concat_df = pd.concat([lt_mean_vc, ge_mean_vc], axis = 0)
    return  concat_df[['index','{}_offset_mean_cum_count'.format(f)]]

def augment(train, num_n=1, num_p=2):
    newtrain = [train]
    n = train[train.target == 0]
    for i in range(num_n):
        newtrain.append(n.apply(lambda x: x.values.take(np.random.permutation(len(n)))))
    for i in range(num_p):
        p = train[train.target > 0]
        newtrain.append(p.apply(lambda x: x.values.take(np.random.permutation(len(p)))))
    return pd.concat(newtrain)

for f in tqdm(features_original):
    uv = train['{}_ID_code_counts'.format(f)].unique().tolist()
    cv = np.min(sorted(list(set(uv))))
    train["cut_{}".format(f)] = train[f]
    median_v =  train[f].median()
    train.loc[train['{}_ID_code_counts'.format(f)] == cv, "cut_{}".format(f)] = median_v#-999

for f in tqdm(features_original):
    mean_dist = np.abs(train[f] - total[f].mean())
    max_dist = np.abs(train[f] - total[f].max())
    min_dist = np.abs(train[f] - total[f].min())
    median_dist =  np.abs(train[f] - total[f].median())
    mode_dist =  np.abs(train[f] - total[f].mode())
    cumsum = train[[f]].merge(get_hist_cum(total, f), left_on = [f], right_on = ['index'], how = 'left')
    train['{}_mean_ratio'.format(f)] = np.log(train['{}_ID_code_counts'.format(f)]) / mean_dist
    train['{}_max_ratio'.format(f)] = np.log(train['{}_ID_code_counts'.format(f)]) / max_dist
    train['{}_min_ratio'.format(f)] =np.log(train['{}_ID_code_counts'.format(f)]) / min_dist
    train['{}_median_ratio'.format(f)] =np.log(train['{}_ID_code_counts'.format(f)]) / median_dist
    train['{}_mode_ratio'.format(f)] =np.log(train['{}_ID_code_counts'.format(f)]) / mode_dist
    train['{}_cumcum_ratio'.format(f)] =np.log(train['{}_ID_code_counts'.format(f)]) / (cumsum['cum_count{}'.format(f)] / len(total))
    train['{}_cumsum_ratio_sub'.format(f)] =np.log(train['{}_ID_code_counts'.format(f)]) / (1 - cumsum['cum_count{}'.format(f)] / len(total))



for f in tqdm(features_original):
    train['{}_ID_code_counts'.format(f)] = minmax_scale(train['{}_ID_code_counts'.format(f)])


test = train[len_train:]
train = train[:len_train]
lgb_params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 6,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 160,
    "tree_learner": "serial",
    "boost_from_average": "false",
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "num_threads": 48,
    "seed": random_state
}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
oof = train[['ID_code', 'target']]
oof['predict'] = 0
predictions = test[['ID_code']]
val_aucs = []

features = [col for col in test.columns if col not in ['target', 'ID_code']]
X_test = test[features].values

for fold, (trn_idx, val_idx) in enumerate(skf.split(train, train['target'])):
    X_train, y_train = train.iloc[trn_idx][features], train.iloc[trn_idx]['target']
    X_valid, y_valid = train.iloc[val_idx][features], train.iloc[val_idx]['target']
    trn_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    evals_result = {}
    lgb_clf = lgb.train(lgb_params,
                        trn_data,
                        100000,
                        valid_sets=[val_data],
                        early_stopping_rounds=3000,
                        verbose_eval=1000,
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

predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)
sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})
sub_df["target"] = predictions['target']
sub_df.to_csv("bao_lgb_sub_{}.csv".format(all_auc), index=False)
oof.to_csv('bao_lgb_oof_{}.csv'.format(all_auc), index=False)
predictions.to_csv('bao_lgb_all_predictions_{}.csv'.format(all_auc), index=False)
predictions[['ID_code', 'target']].to_csv('submission.csv', index=False)
