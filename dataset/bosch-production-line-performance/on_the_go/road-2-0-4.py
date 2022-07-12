# -*- coding: utf-8 -*-
"""
@author: Faron
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import gc

DATA_DIR = "../input"

ID_COLUMN = 'Id'
TARGET_COLUMN = 'Response'

SEED = 0
CHUNKSIZE = 50000
NROWS = 250000

TRAIN_NUMERIC = "{0}/train_numeric.csv".format(DATA_DIR)
TRAIN_DATE = "{0}/train_date.csv".format(DATA_DIR)

TEST_NUMERIC = "{0}/test_numeric.csv".format(DATA_DIR)
TEST_DATE = "{0}/test_date.csv".format(DATA_DIR)

FILENAME = "etimelhoods"

train = pd.read_csv(TRAIN_NUMERIC, usecols=[ID_COLUMN, TARGET_COLUMN])
test = pd.read_csv(TEST_NUMERIC, usecols=[ID_COLUMN])

train["StartTime"] = -1
test["StartTime"] = -1

print(len(train.Id.unique()))
nrows = 0
for tr, te in zip(pd.read_csv(TRAIN_DATE, chunksize=CHUNKSIZE), pd.read_csv(TEST_DATE, chunksize=CHUNKSIZE)):
    feats = np.setdiff1d(tr.columns, [ID_COLUMN])

    stime_tr = tr[feats].min(axis=1).values
    stime_te = te[feats].min(axis=1).values

    print(len(train.loc[train.Id.isin(tr.Id), 'StartTime']), len(stime_tr))
    train.loc[train.Id.isin(tr.Id), 'StartTime'] = stime_tr
    test.loc[test.Id.isin(te.Id), 'StartTime'] = stime_te

    # nrows += CHUNKSIZE
    # if nrows >= NROWS:
    #     break
    gc.collect()
del tr, te


ntrain = train.shape[0]
train_test = pd.concat((train, test)).reset_index(drop=True).reset_index(drop=False)
# print(train_test)

train_test['0_¯\_(ツ)_/¯_1'] = train_test[ID_COLUMN].diff().fillna(9999999).astype(int)
train_test['0_¯\_(ツ)_/¯_2'] = train_test[ID_COLUMN].iloc[::-1].diff().fillna(9999999).astype(int)

train_test = train_test.sort_values(by=['StartTime', 'Id'], ascending=True)
# print(train_test)

train_test['0_¯\_(ツ)_/¯_3'] = train_test[ID_COLUMN].diff().fillna(9999999).astype(int)
train_test['0_¯\_(ツ)_/¯_4'] = train_test[ID_COLUMN].iloc[::-1].diff().fillna(9999999).astype(int)

train_test = train_test.sort_values(by=['index']).drop(['index'], axis=1)
train_test.to_csv('magic_featrues.csv')
print(train_test.head())
features = np.setdiff1d(list(train.columns), [TARGET_COLUMN, ID_COLUMN])
# features = ['0_¯\_(ツ)_/¯_1', '0_¯\_(ツ)_/¯_2','0_¯\_(ツ)_/¯_3','0_¯\_(ツ)_/¯_4']
y = train.Response.ravel()
train = np.array(train[features])

# print('train: {0}'.format(train.shape))
prior = np.sum(y) / (1.*len(y))

# xgb_params = {
#     'seed': 0,
#     'colsample_bytree': 0.7,
#     'silent': 1,
#     'subsample': 0.7,
#     'learning_rate': 0.1,
#     'objective': 'binary:logistic',
#     'max_depth': 4,
#     'num_parallel_tree': 1,
#     'min_child_weight': 2,
#     'eval_metric': 'auc',
#     'base_score': prior
# }


# dtrain = xgb.DMatrix(train, label=y)
# bst = xgb.train(xgb_params, dtrain, num_boost_round=10)
# del train

# test = train_test.iloc[ntrain:, :]
# print(test.head())
# test = np.array(test[features])

# dtest = xgb.DMatrix(test)
# preds = bst.predict(dtest)
# print(preds)
# labels = dtest.get_label()

# Id = train_test.iloc[ntrain:, 0].values
# print(Id)
# print(type(Id))
# print(labels)
# print(type(labels))

X_train = train_test.iloc[:ntrain, 3:7]
y_train = train_test.iloc[:ntrain, 1]
clf = xgb.XGBClassifier(seed=0, silent=1, learning_rate=0.7, objective='binary:logistic',
max_depth=4, min_child_weight=2, base_score=prior)
clf.fit(X_train, y_train, eval_metric='auc')
X_test = train_test.iloc[ntrain:, 3:7]
y_test = clf.predict(X_test)
print(type(y_test))

Id = train_test.iloc[ntrain:, 0].values
print(len(Id))
result = pd.DataFrame(y_test, index=Id)
print(len(result))
result.to_csv('submission.csv', header=False)



# result = pd.DataFrame(labels, index=Id)
# result.to_csv('submission.csv', header=False)

# res = xgb.cv(xgb_params, dtrain, num_boost_round=10, nfold=4, seed=0, stratified=True,
#              early_stopping_rounds=1, verbose_eval=1, show_stdv=True)

# cv_mean = res.iloc[-1, 0]
# cv_std = res.iloc[-1, 1]

# print('CV-Mean: {0}+{1}'.format(cv_mean, cv_std))


