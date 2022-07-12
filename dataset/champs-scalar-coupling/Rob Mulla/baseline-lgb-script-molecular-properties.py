import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pylab as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
structures = pd.read_csv('../input/structures.csv')

#####################
## FEATURE CREATION
#####################

def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

train_df = map_atom_info(train_df, 0)
train_df = map_atom_info(train_df, 1)

test_df = map_atom_info(test_df, 0)
test_df = map_atom_info(test_df, 1)

# https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark
train_p_0 = train_df[['x_0', 'y_0', 'z_0']].values
train_p_1 = train_df[['x_1', 'y_1', 'z_1']].values
test_p_0 = test_df[['x_0', 'y_0', 'z_0']].values
test_p_1 = test_df[['x_1', 'y_1', 'z_1']].values

train_df['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
test_df['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)

# make categorical variables
atom_map = {'H': 0,
            'C': 1,
            'N': 2}
train_df['atom_0_cat'] = train_df['atom_0'].map(atom_map).astype('int')
train_df['atom_1_cat'] = train_df['atom_1'].map(atom_map).astype('int')
test_df['atom_0_cat'] = test_df['atom_0'].map(atom_map).astype('int')
test_df['atom_1_cat'] = test_df['atom_1'].map(atom_map).astype('int')

# One Hot Encode the Type
train_df = pd.concat([train_df, pd.get_dummies(train_df['type'])], axis=1)
test_df = pd.concat([test_df, pd.get_dummies(test_df['type'])], axis=1)

train_df['dist_to_type_mean'] = train_df['dist'] / train_df.groupby('type')['dist'].transform('mean')
test_df['dist_to_type_mean'] = test_df['dist'] / test_df.groupby('type')['dist'].transform('mean')

# Atom Count
atom_count_dict = structures.groupby('molecule_name').count()['atom_index'].to_dict()
train_df['atom_count'] = train_df['molecule_name'].map(atom_count_dict)
test_df['atom_count'] = test_df['molecule_name'].map(atom_count_dict)

#####################
## CONFIGURABLES
#####################

FEATURES = ['atom_index_0', 'atom_index_1',
            'atom_0_cat',
            'x_0', 'y_0', 'z_0',
            'atom_1_cat', 
            'x_1', 'y_1', 'z_1', 'dist', 'dist_to_type_mean',
            'atom_count',
            '1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN'
           ]
TARGET = 'scalar_coupling_constant'
CAT_FEATS = ['atom_0','atom_1']
N_ESTIMATORS = 11000
VERBOSE = 500
EARLY_STOPPING_ROUNDS = 200
RANDOM_STATE = 529

#####################
## CREATE FINAL DATASETS
#####################

X = train_df[FEATURES]
X_test = test_df[FEATURES]
y = train_df[TARGET]

#####################
## TRAIN MODEL
#####################

lgb_params = {'num_leaves': 128,
              'min_child_samples': 64,
              'objective': 'regression',
              'max_depth': 6,
              'learning_rate': 0.3,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 0.9,
              "bagging_seed": 11,
              "metric": 'mae',
              "verbosity": -1,
              'reg_alpha': 0.1,
              'reg_lambda': 0.4,
              'colsample_bytree': 1.0
         }

n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=RANDOM_STATE)

# Setup arrays for storing results
oof = np.zeros(len(X))
prediction = np.zeros(len(X_test))
scores = []
feature_importance = pd.DataFrame()

# Train the model
for fold_n, (train_idx, valid_idx) in enumerate(folds.split(X)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    model = lgb.LGBMRegressor(**lgb_params, n_estimators = N_ESTIMATORS, n_jobs = -1)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_valid, y_valid)],
              eval_metric='mae',
              verbose=VERBOSE,
              early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    print('predicting valid')
    y_pred_valid = model.predict(X_valid)
    print('predicting test')
    y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
    
    # feature importance
    print('Saving fold importance')
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = FEATURES
    fold_importance["importance"] = model.feature_importances_
    fold_importance["fold"] = fold_n + 1
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
    print('Saving fold score')
    scores.append(mean_absolute_error(y_valid, y_pred_valid))
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    print('Saving oof')
    oof[valid_idx] = y_pred_valid.reshape(-1,)
    print('appending fold predictions')
    prediction += y_pred

prediction /= folds.n_splits
    
#####################
# SAVE RESULTS
#####################

# Save Prediction and name appropriately
submission_csv_name = 'submission_lgb_{}folds_{}CV.csv'.format(n_fold, np.mean(scores))
oof_csv_name = 'oof_lgb_{}folds_{}CV.csv'.format(n_fold, np.mean(scores))
fi_csv_name = 'fi_lgb_{}folds_{}CV.csv'.format(n_fold, np.mean(scores))

print('Saving LGB Submission as:')
print(submission_csv_name)
ss = pd.read_csv('../input/sample_submission.csv')
ss['scalar_coupling_constant'] = prediction
ss.to_csv(submission_csv_name, index=False)
ss.head()
# OOF
oof_df = train_df[['id','molecule_name','scalar_coupling_constant']].copy()
oof_df['oof_pred'] = oof
oof_df.to_csv(oof_csv_name, index=False)
# Feature Importance
feature_importance.to_csv(fi_csv_name, index=False)