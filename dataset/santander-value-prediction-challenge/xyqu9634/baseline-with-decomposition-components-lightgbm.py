# -*- coding: utf-8 -*-
# """
# Created on Wed Jun 20 10:21:18 2018

# @author: qxy

# from Vladimir Demidov's Baseline with decomposition components using lightgbm
# https://www.kaggle.com/yekenot/baseline-with-decomposition-components

# """

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

print("Load data...")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')
# train = pd.read_csv('data/train.csv')
# test = pd.read_csv('data/test.csv')
# subm = pd.read_csv('data/sample_submission.csv')
print("Train shape: {}\nTest shape: {}".format(train.shape, test.shape))
#%%
PERC_TRESHOLD = 0.99   ### Percentage of zeros in each feature ###
          ### Number of decomposition components ###


target = np.log1p(train['target']).values
cols_to_drop = [col for col in train.columns[2:]
                    if [i[1] for i in list(train[col].value_counts().items()) 
                    if i[0] == 0][0] >= train.shape[0] * PERC_TRESHOLD]

print("Define training features...")
exclude_other = ['ID', 'target']
train_features = []
for c in train.columns:
    if c not in cols_to_drop \
    and c not in exclude_other:
        train_features.append(c)
print("Number of featuress for training: %s" % len(train_features))

train, test = train[train_features], test[train_features]

print("\nTrain shape: {}\nTest shape: {}".format(train.shape, test.shape))

N_COMP = 40  

print("\nStart decomposition process...")
print("PCA")
pca = PCA(n_components=N_COMP, random_state=17)
pca_results_train = pca.fit_transform(train)
pca_results_test = pca.transform(test)

print("tSVD")
tsvd = TruncatedSVD(n_components=N_COMP, random_state=17)
tsvd_results_train = tsvd.fit_transform(train)
tsvd_results_test = tsvd.transform(test)

print("ICA")
ica = FastICA(n_components=N_COMP, random_state=17)
ica_results_train = ica.fit_transform(train)
ica_results_test = ica.transform(test)

print("GRP")
grp = GaussianRandomProjection(n_components=N_COMP, eps=0.1, random_state=17)
grp_results_train = grp.fit_transform(train)
grp_results_test = grp.transform(test)

print("SRP")
srp = SparseRandomProjection(n_components=N_COMP, dense_output=True, random_state=17)
srp_results_train = srp.fit_transform(train)
srp_results_test = srp.transform(test)

decomposition_features = []

print("Append decomposition components to datasets...")
for i in range(1, N_COMP + 1):
    train['pca_' + str(i)] = pca_results_train[:, i - 1]
    test['pca_' + str(i)] = pca_results_test[:, i - 1]
    decomposition_features.append('pca_' + str(i))

    train['ica_' + str(i)] = ica_results_train[:, i - 1]
    test['ica_' + str(i)] = ica_results_test[:, i - 1]
    decomposition_features.append('ica_' + str(i))

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]
    decomposition_features.append('tsvd_' + str(i))

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]
    decomposition_features.append('grp_' + str(i))

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]
    decomposition_features.append('srp_' + str(i))
    
print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))
train = train.values


print('\nModelling...')

N = 5
kf = KFold(n_splits=N,shuffle=False,random_state=42)
cv_scores = []
cv_predictions = []

for train_in,test_in in kf.split(train, target):
    X_train, X_valid, y_train, y_valid = train[train_in], train[test_in], target[train_in], target[test_in]

    lgb_model = LGBMRegressor(
            n_estimators=10000,
            # num_leaves=30,
            # colsample_bytree=.8,
            # subsample=.9,
            # max_depth=7,
            # reg_alpha=.1,
            # reg_lambda=.1,
            # min_split_gain=.01
            learning_rate=0.01,
            )

    lgb_model.fit(X_train, y_train,
                 eval_set=[(X_valid, y_valid)],
                 eval_metric='rmse', verbose=100, early_stopping_rounds=80)

    score = lgb_model.best_score_['valid_0']['rmse']
    cv_scores.append(score)
    
    y_pred = lgb_model.predict(test)
    cv_predictions.append(y_pred)

s = 0
for i in cv_predictions:
    s = s + i
y_preds = s /N

p_score = np.mean(cv_scores)
subm['target'] = np.exp(y_preds)-1
subm.to_csv('submit_%s_%s_%s.csv'%(N, N_COMP, round(p_score,5)), index=False)