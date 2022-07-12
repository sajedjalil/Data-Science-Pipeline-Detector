"""
Based on the Forza Baseline (https://www.kaggle.com/the1owl/forza-baseline)
kernel and "A Kaggler's Guide to Model Stacking in Practice"
(http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/)
article.
"""

import numpy as np
import pandas as pd
import xgboost
import lightgbm

from sklearn.model_selection import StratifiedKFold, train_test_split,\
    GridSearchCV
from sklearn.metrics import auc, roc_curve, make_scorer
from sklearn.linear_model import LogisticRegression

import gc


train_data = pd.read_csv("../input/train.csv", index_col='id', na_values='-1')
test_data = pd.read_csv("../input/test.csv", index_col='id',
                        na_values=['-1', '-1.0'])

full_data = pd.concat([train_data, test_data])

for c in full_data.columns:
    if c.endswith('cat'):
        full_data[c] = full_data[c].astype('category')

for c in full_data.columns:
    if c.startswith('ps_calc'):
        full_data.drop(c, axis=1, inplace=True)

for c in full_data.drop('target', axis=1).columns:
    if (2 < len(list(full_data[c].unique())) < 7 and
            full_data[c].dtype.name != 'category'):
        print("Column '%s' has less than %i unique values, converting "
              "it to a category variable." % (c, len(full_data[c].unique())))
        full_data[c] = full_data[c].astype('category')

full_data['ps_car_13_x_ps_reg_03'] =\
    full_data['ps_car_13'] * full_data['ps_reg_03']

full_data['number_of_nans_in_row'] = full_data.isnull().sum(axis=1)

full_data = pd.get_dummies(full_data)

train_data = full_data.iloc[:train_data.shape[0]]
test_data = full_data.iloc[train_data.shape[0]:]
test_data.drop('target', axis=1, inplace=True)

del full_data
gc.collect()

train_meta = train_data.copy()
train_meta['xgboost_prediction'] = np.nan
train_meta['lightgbm_prediction'] = np.nan


def assert_no_leackage(dataset):
    for c in dataset.columns:
        assert c != 'target'
        assert not c.startswith('target_')
        assert not c.startswith('ps_calc')


assert_no_leackage(train_data.drop('target', axis=1))


def add_greater_than_mean_and_median_cols(dataset):
    for c in dataset.columns:
        if (not c.endswith('_bin') and
                dataset[c].dtype.name != 'category' and
                c != 'target'):
            dataset[c + '_greater_than_mean'] =\
                dataset[c] > dataset[c].mean()
            dataset[c + '_greater_than_median'] =\
                dataset[c] > dataset[c].median()
    return dataset


def gini(y, pred):
    fpr, tpr, thr = roc_curve(y, pred, pos_label=1)
    g = 2 * auc(fpr, tpr) - 1
    return g


def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred)


def gini_lgbm(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True


skf = StratifiedKFold(n_splits=5, random_state=0)

lightgbm_params = {
    'learning_rate': 0.02,
    'max_depth': 4,
    'boosting': 'gbdt',
    'objective': 'binary',
    'max_bin': 10,
    'subsample': 0.8,
    'subsample_freq': 10,
    'colsample_bytree': 0.8,
    'min_child_samples': 500,
    'metric': 'auc',
    'is_training_metric': False,
    'seed': 0
}

counter = 0
for train_index, test_index in skf.split(train_data.drop('target', axis=1),
                                         train_data.target):
    print("--- LightGBM fold #%i ---" % (counter + 1))
    X_train, y_train =\
        train_data.drop('target', axis=1).iloc[train_index],\
        train_data.target.iloc[train_index]
    X_test, y_test =\
        train_data.drop('target', axis=1).iloc[test_index],\
        train_data.target.iloc[test_index]
    X_train = add_greater_than_mean_and_median_cols(X_train)
    X_test = add_greater_than_mean_and_median_cols(X_test)
    assert_no_leackage(X_train)
    assert_no_leackage(X_test)
    lightgbm_model = lightgbm.train(
        lightgbm_params,
        train_set=lightgbm.Dataset(X_train, label=y_train),
        num_boost_round=3000,
        valid_sets=lightgbm.Dataset(X_test, label=y_test),
        verbose_eval=50,
        feval=gini_lgbm,
        early_stopping_rounds=200)
    train_meta['lightgbm_prediction'].iloc[test_index] =\
        lightgbm_model.predict(
            X_test,
            num_iteration=lightgbm_model.best_iteration)
    counter += 1

train_meta['lightgbm_prediction'] =\
    (np.exp(train_meta['lightgbm_prediction'].values) - 1.0).clip(0, 1)

xgboost_params = {
    'eta': 0.02,
    'max_depth': 4,
    'objective': 'binary:logistic',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 0.77,
    'scale_pos_weight': 1.6,
    'gamma': 10,
    'reg_alpha': 8,
    'reg_lambda': 1.3,
    'eval_metric':
    'auc',
    'seed': 99,
    'silent': True
}

counter = 0
for train_index, test_index in skf.split(train_data.drop('target', axis=1),
                                         train_data.target):
    print("--- XGBoost fold #%i ---" % (counter + 1))
    X_train, y_train =\
        train_data.drop('target', axis=1).iloc[train_index],\
        train_data.target.iloc[train_index]
    X_test, y_test =\
        train_data.drop('target', axis=1).iloc[test_index],\
        train_data.target.iloc[test_index]
    X_train = add_greater_than_mean_and_median_cols(X_train)
    X_test = add_greater_than_mean_and_median_cols(X_test)
    assert_no_leackage(X_train)
    assert_no_leackage(X_test)
    watchlist = [(xgboost.DMatrix(X_train, y_train), 'train'),
                 (xgboost.DMatrix(X_test, y_test), 'valid')]
    xgboost_model = xgboost.train(
        xgboost_params,
        xgboost.DMatrix(X_train, y_train),
        5000,
        watchlist,
        feval=gini_xgb,
        maximize=True,
        verbose_eval=50,
        early_stopping_rounds=200)
    train_meta['xgboost_prediction'].iloc[test_index] =\
        xgboost_model.predict(
            xgboost.DMatrix(X_test),
            ntree_limit=xgboost_model.best_ntree_limit + 45)
    counter += 1

train_meta['xgboost_prediction'] =\
    (np.exp(train_meta['xgboost_prediction'].values) - 1.0).clip(0, 1)

train_meta.to_csv("train_meta.csv")


def scorer_helper(y, y_pred):
    return gini(y, y_pred[:, 1])


scorer = make_scorer(scorer_helper, needs_proba=True)

param_grid = {
    'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
}
stacker = GridSearchCV(LogisticRegression(max_iter=5000, random_state=0),
                       param_grid, scoring=scorer, cv=skf)
stacker.fit(train_meta[['lightgbm_prediction', 'xgboost_prediction']],
            train_meta.target)

train_data = add_greater_than_mean_and_median_cols(train_data)
test_data = add_greater_than_mean_and_median_cols(test_data)

assert_no_leackage(train_data.drop('target', axis=1))
assert_no_leackage(test_data)

test_meta = pd.DataFrame(
    {
        'lightgbm_prediction': np.nan,
        'xgboost_prediction': np.nan,
    },
    index=test_data.index
)

x1, x2, y1, y2 =\
    train_test_split(train_data.drop('target', axis=1),
                     train_data.target, test_size=0.25, random_state=0)

assert_no_leackage(x1)
assert_no_leackage(x2)

print("Train the LightGBM model on the full train dataset...")

lightgbm_model = lightgbm.train(
    lightgbm_params,
    train_set=lightgbm.Dataset(
        x1, label=y1),
    valid_sets=lightgbm.Dataset(x2, label=y2),
    num_boost_round=3000,
    verbose_eval=50,
    feval=gini_lgbm,
    early_stopping_rounds=200)

print("Saving LightGBM model...")

lightgbm_model.save_model('lightgbm.model')

test_meta['lightgbm_prediction'] = (np.exp(lightgbm_model.predict(
    test_data, num_iteration=lightgbm_model.best_iteration)) - 1.0).clip(0, 1)

del lightgbm_model
gc.collect()

print("Train the XGBoost model on the full train dataset...")

watchlist = [(xgboost.DMatrix(x1, y1), 'train'),
             (xgboost.DMatrix(x2, y2), 'valid')]

xgboost_model = xgboost.train(
    xgboost_params,
    xgboost.DMatrix(x1, y1),
    5000,
    watchlist,
    feval=gini_xgb,
    maximize=True,
    verbose_eval=50,
    early_stopping_rounds=200)

print("Saving XGBoost model...")

xgboost_model.save_model('xgboost.model')

test_meta['xgboost_prediction'] = (np.exp(
    xgboost_model.predict(
        xgboost.DMatrix(test_data),
        ntree_limit=xgboost_model.best_ntree_limit + 45)) - 1.0).clip(0, 1)

del xgboost_model
gc.collect()

test_meta.to_csv("test_meta.csv")

stacker_prediction = (np.exp(
    stacker.predict_proba(test_meta)[:, 1]) - 1.0).clip(0, 1)
stacked_submission = pd.DataFrame({'target': stacker_prediction},
                                  index=test_data.index)
stacked_submission.to_csv("stacked_submission.csv")