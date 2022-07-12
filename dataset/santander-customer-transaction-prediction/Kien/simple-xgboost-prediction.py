# Simple XGBoost prediction.
# I didn't tune both of feature and hyperparameter.
import datetime
import gc
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import warnings

from contextlib import contextmanager
from pandas.core.common import SettingWithCopyWarning
from sklearn.model_selection import KFold

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

target = train['target']
train_df = train.drop(['ID_code', 'target'],axis = 1)
test_df  = test.drop(['ID_code'],axis = 1)
feats = train_df.columns

fold_xgb = KFold(n_splits=2, shuffle=False, random_state=114)
def kfold_xgboost(train_df, test_df, debug = False):
    oof_preds_xgb = np.zeros(train_df.shape[0])
    sub_preds_xgb = np.zeros(test_df.shape[0])
    xgb_params = {
    'objective': 'binary:logitraw',
    'eval_metric': 'auc',
    'booster': 'gbtree',
    'n_jobs': 4,
    'tree_method': 'hist',
    'eta': 0.01,
    'grow_policy': 'lossguide',
    'max_delta_step': 2,
    'seed': 538,
    'colsample_bylevel': 0.9,
    'colsample_bytree': 0.8,
    'gamma': 1.0,
    'learning_rate': 0.001,
    'max_bin': 64,
    'max_depth': 6,
    'max_leaves': 10,
    'min_child_weight': 10,
    'reg_alpha': 1e-06,
    'reg_lambda': 1.0,
    'subsample': 0.7}
    for fold_, (train_idx, valid_idx) in enumerate(fold_xgb.split(train_df.values)):
        train_x, train_y = train_df.iloc[train_idx], train['target'].iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx], train['target'].iloc[valid_idx]
        print("fold n °{}".format(fold_))
        trn_Data = xgb.DMatrix(train_x, label = train_y)
        val_Data = xgb.DMatrix(valid_x, label = valid_y)
        watchlist = [(trn_Data, "Train"), (val_Data, "Valid")]
        print("xgb" + str(fold_) + "-" * 50)
        num_rounds = 10000
        xgb_model = xgb.train(xgb_params, trn_Data,num_rounds,watchlist,early_stopping_rounds=50, verbose_eval= 1000)
        oof_preds_xgb[valid_idx] = xgb_model.predict(xgb.DMatrix(train_df.iloc[valid_idx][feats]), ntree_limit = xgb_model.best_ntree_limit + 50)
        sub_preds_xgb = xgb_model.predict(xgb.DMatrix(test_df[feats]),ntree_limit= xgb_model.best_ntree_limit)/fold_xgb.n_splits
        
        del train_idx,valid_idx
        gc.collect()
    xgb.plot_importance(xgb_model)
    plt.figure(figsize = (16,10))
    plt.savefig("importance.png")
    #xgb.to_graphviz(xgb_model)
    return sub_preds_xgb

def main():
    submission = pd.read_csv("../input/sample_submission.csv")
    Preds_xgb = kfold_xgboost(train_df, test_df, debug = False)
    submission['target'] = Preds_xgb
    submission.to_csv(submission_name, index = False)

if __name__ == "__main__":
    submission_name = "submission.csv"
    with timer("Model Finished"):
        main()

    
