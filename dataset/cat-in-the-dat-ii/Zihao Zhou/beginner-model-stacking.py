# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import time
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
import warnings as wrn
wrn.filterwarnings('ignore', category=DeprecationWarning)
wrn.filterwarnings('ignore', category=FutureWarning)
wrn.filterwarnings('ignore', category=UserWarning)

#Load the data
train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
target, train_id = train['target'], train['id']
test_id = test['id']
train.drop(['id', 'target'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

#Fill the absent value
def replace_nan(data):
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].mode()[0])

replace_nan(train)
replace_nan(test)

#Feature labelling for binary features
bin_map = {'F':0, 'T':1, 'N':0, 'Y':1}
train['bin_3'] = train['bin_3'].map(bin_map)
train['bin_4'] = train['bin_4'].map(bin_map)

test['bin_3'] = test['bin_3'].map(bin_map)
test['bin_4'] = test['bin_4'].map(bin_map)

#Feature labelling for nominal features(relatively low cardinalty)
nom_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']
onehot_train_nom = pd.get_dummies(train[nom_cols])
onehot_test_nom = pd.get_dummies(test[nom_cols])
train.drop(labels= nom_cols, axis=1, inplace=True)
test.drop(labels=nom_cols, axis=1,  inplace=True)
train = pd.concat([train, onehot_train_nom], axis=1)
test = pd.concat([test, onehot_test_nom], axis=1)

#Feature target labelling for nominal features with high cardinalty
#Notice that, the target encoding should be used in a cross-validate framework to decrease the effect of target data leakage
high_cardinality_cols = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
kf_encoder = StratifiedKFold(n_splits=5, random_state=0)
val_df = pd.DataFrame([])
for train_index, val_index in kf_encoder.split(train, target):
    target_encoder = TargetEncoder(cols = high_cardinality_cols, smoothing=0.20)
    target_encoder.fit(train.iloc[train_index, :], target[train_index])
    val_df = val_df.append(target_encoder.transform(train.iloc[val_index, :]), ignore_index=False)

test_target_encoder = TargetEncoder(cols=high_cardinality_cols, smoothing=0.20)
test_target_encoder.fit(train, target)
train[high_cardinality_cols] = val_df[high_cardinality_cols].sort_index()
test[high_cardinality_cols] = test_target_encoder.transform(test)[high_cardinality_cols]

#Feature labelling for ordered features
ord_order = [
    [1.0, 2.0, 3.0],
    ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'],
    ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']
]

for i in range(1, 3):
    ord_order_dict = {i : j for j, i in enumerate(ord_order[i])}
    train[f'ord_{i}'] = train[f'ord_{i}'].map(ord_order_dict)
    test[f'ord_{i}'] = test[f'ord_{i}'].map(ord_order_dict)

for i in range(3, 6):
    ord_order_dict = {i : j for j, i in enumerate(sorted(list(set(list(train[f'ord_{i}'].unique()) + list(test[f'ord_{i}'].unique())))))}
    train[f'ord_{i}'] = train[f'ord_{i}'].map(ord_order_dict)
    test[f'ord_{i}'] = test[f'ord_{i}'].map(ord_order_dict)

#Feature scaling
my_standard = StandardScaler()
train = my_standard.fit_transform(train)
test = my_standard.transform(test)

#Prepare for the model training
n_folds = 5
kf_train = StratifiedKFold(n_splits=n_folds, random_state=0)
def cross_validate_train(model, model_name, train, target, test):
    pred_train = np.zeros((train.shape[0],1))
    pred_test = np.zeros((n_folds, test.shape[0]))
    start = time.time()
    for i, (train_index, val_index) in enumerate(kf_train.split(train, target)):
        train_X, val_X = train[train_index, :], train[val_index,:]
        train_y, val_y = target[train_index], target[val_index]
        if model_name == 'CatBoost':
            model.fit(train_X, train_y, early_stopping_rounds=50,
                      plot=False)
        else:
            model.fit(train_X, train_y)
        pred_current_val = model.predict_proba(val_X)[:,1]
        pred_train[val_index] = pred_current_val.reshape(-1,1)
        pred_test[i] = model.predict_proba(test)[:,1]
    pred_mean_score = roc_auc_score(target, pred_train)
    final_pred_test = pred_test.mean(axis=0)
    end = time.time()
    print('Total training Time for {}: {}'.format(model_name, end-start))
    print('Final mean cv score: {}'.format(pred_mean_score))
    return [pred_train.reshape(-1,1), final_pred_test.reshape(-1,1)]


model_lr = LogisticRegression(C = 0.01, n_jobs=-1, max_iter=500, random_state=0, fit_intercept=True, penalty='none')
model_sgd = SGDClassifier(loss = 'log', n_jobs=-1, random_state=0, early_stopping=True, n_iter_no_change=10)
model_lgb = LGBMClassifier(learning_rate=0.02, n_estimators=1000, random_state=0, n_jobs=-1)
model_cat = CatBoostClassifier(bagging_temperature = 0.8, depth = 6, iterations=1000,
                               l2_leaf_reg=30, learning_rate=0.05, random_strength=0.8,
                               loss_function='Logloss', eval_metric = 'AUC', thread_count=2,
                               verbose = False)

#First layer model training

train_lr, pred_lr = cross_validate_train(model_lr, 'LogisticRegression', train, target, test)
train_sgd, pred_sgd = cross_validate_train(model_sgd, 'SGD-SVC', train, target, test)
train_lgb, pred_lgb = cross_validate_train(model_lgb, 'Lightgbm', train, target, test)
train_cat, pred_cat = cross_validate_train(model_cat, 'CatBoost', train, target, test)

x_train = np.concatenate((train_lr, train_sgd, train_lgb, train_cat), axis=1)
x_test = np.concatenate((pred_lr, pred_sgd, pred_lgb, pred_cat), axis=1)

#Final Xgboost model training
kf = StratifiedKFold(n_splits=n_folds, random_state=0)
final_xgb_model = XGBClassifier(max_depth=3, n_estimators=1000, learning_rate=0.01, n_jobs=-1,
                                random_state=0, objective ='binary:logistic',
                                subsample = 0.8, verbosity=0)
xgb_pred_train = np.zeros((x_train.shape[0], 1))
xgb_pred_test = np.zeros((n_folds, x_test.shape[0]))
for i,(train_index, val_index) in enumerate(kf.split(x_train, target)):
    X_train, X_val = x_train[train_index,:], x_train[val_index,:]
    y_train, y_val = target[train_index], target[val_index]
    final_xgb_model.fit(X_train, y_train)
    current_val_pred = final_xgb_model.predict_proba(X_val)[:,1]
    current_cv_score = roc_auc_score(y_val, current_val_pred)
    print('Folds:{} , cv_score:{}'.format(i, current_cv_score))
    xgb_pred_train[val_index] = current_val_pred.reshape(-1,1)
    xgb_pred_test[i] = final_xgb_model.predict_proba(x_test)[:,1]

final_mean_score = roc_auc_score(target, xgb_pred_train)
print('Final score for ensemble learning: {}'.format(final_mean_score))
final_pred_test = xgb_pred_test.mean(axis=0)

"""
kf = StratifiedKFold(n_splits=n_folds, random_state=0)
final_rf_model = RandomForestClassifier(max_depth=3, n_estimators=1000, n_jobs=-1,
                                random_state=0)
rf_pred_train = np.zeros((x_train.shape[0], 1))
rf_pred_test = np.zeros((n_folds, x_test.shape[0]))
for i,(train_index, val_index) in enumerate(kf.split(x_train, target)):
    X_train, X_val = x_train[train_index,:], x_train[val_index,:]
    y_train, y_val = target[train_index], target[val_index]
    final_rf_model.fit(X_train, y_train)
    current_val_pred = final_rf_model.predict_proba(X_val)[:,1]
    current_cv_score = roc_auc_score(y_val, current_val_pred)
    print('Folds:{} , cv_score:{}'.format(i, current_cv_score))
    rf_pred_train[val_index] = current_val_pred.reshape(-1,1)
    rf_pred_test[i] = final_rf_model.predict_proba(x_test)[:,1]

final_mean_score = roc_auc_score(target, rf_pred_train)
print('Final score for ensemble learning: {}'.format(final_mean_score))
final_pred_test = rf_pred_test.mean(axis=0)
"""

result = pd.DataFrame({'id' : test_id,
                       'target' : final_pred_test})
result.to_csv('self_submission.csv', index=False)