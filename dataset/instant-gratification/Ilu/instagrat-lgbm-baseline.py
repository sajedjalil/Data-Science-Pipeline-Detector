import gc
import os
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import gc
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

np.random.random(42)
NFOLDS = 6
scaler = 'standard'

print('Reading Train\n')
train = pd.read_csv("../input/train.csv", index_col='id') 
print('Reading Test\n')
test = pd.read_csv("../input/test.csv", index_col='id')
submit = pd.read_csv('../input/sample_submission.csv')

def normal(train, test):
    print('Scaling with StandardScaler\n')
    len_train = len(train)

    traintest = pd.concat([train,test], axis=0, ignore_index=True).reset_index(drop=True)
    
    scaler = StandardScaler()
    cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
    traintest[cols] = scaler.fit_transform(traintest[cols])
    train = traintest[:len_train].reset_index(drop=True)
    test = traintest[len_train:].reset_index(drop=True)

    return train, test

def run_model(train_df, test_df):
    
    features = [c for c in train_df.columns if c not in ['id', 'target']]
    target = train_df['target']
    
    train_df['wheezy-copper-turtle-magic'] = train_df['wheezy-copper-turtle-magic'].astype('category')
    test_df['wheezy-copper-turtle-magic'] = test_df['wheezy-copper-turtle-magic'].astype('category')
    
    param = {
        #'bagging_freq': 5,
        #'bagging_fraction': 0.8,
        'bagging_seed': 0,
        'boost_from_average':'true',
        'boost': 'gbdt',
        'feature_fraction': 0.9,
        'learning_rate': 0.1,
        'max_depth': -1,
        'metric':'auc',
        'min_data_in_leaf': 200,
        'min_sum_hessian_in_leaf': 10, 
        'num_leaves': 38,
        'num_threads': 8,
        'tree_learner': 'serial',
        'objective': 'binary', 
        'verbosity': 1,
        'reg_alpha': 0.1,
        #'reg_lambda': 0.3,
    }
    num_round = 5000
    folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    oof = np.zeros(len(train_df))
    predictions = np.zeros(len(test_df))
    feature_importance = pd.DataFrame()
    
    #features = ['wheezy-copper-turtle-magic']
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
        print("Fold {}".format(fold_))
        trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
        val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])
        clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)
        oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
        predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits
        
        # feature importance
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = features
        fold_importance["importance"] = clf.feature_importance(importance_type='gain')
        fold_importance["fold"] = fold_ + 1
        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
        
    feature_importance["importance"] /= NFOLDS
    cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:].index
    
    best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
    
    plt.figure(figsize=(20, 30));
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
    plt.title('LGB Features (avg over folds)');
    plt.show()
    plt.savefig('lgbm_importances.png')

    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
    return predictions

def submit(predictions):
    submit = pd.read_csv('../input/sample_submission.csv')
    submit["target"] = predictions
    submit.to_csv("submission.csv", index=False)

if scaler == 'standard':
    train, test = normal(train, test)

predictions = run_model(train,test)
submit(predictions)