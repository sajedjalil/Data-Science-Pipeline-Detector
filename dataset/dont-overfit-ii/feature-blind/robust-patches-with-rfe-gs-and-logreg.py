# this is a variation of my previous kernel, but:
# - using LogisticRegression
# - not using noise
# - with a few features added

# This code yields a bit different result when run locally on my laptop.
# The first iteration picks a smaller subset of features and in result it is taken to the
# ensemble. Then 9/20 models are ensembled and Public LB of 0.868 is achieved.

import sys, os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, r2_score, make_scorer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def with_statistics(X):
    statistics = pd.DataFrame()
    statistics['mean']   = X.mean(axis=1)
    statistics['kurt']   = X.kurt(axis=1)
    statistics['mad']    = X.mad(axis=1)
    statistics['median'] = X.median(axis=1)
    statistics['max']    = X.max(axis=1)
    statistics['min']    = X.min(axis=1)
    statistics['skew']   = X.skew(axis=1)
    statistics['sem']    = X.sem(axis=1)
    
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(5, n_jobs=-1)
    neigh.fit(X)

    dists, _ = neigh.kneighbors(X, n_neighbors=5)
    dists = np.delete(dists, 0, 1)
    statistics['minDist'] = dists.mean(axis=1)
    statistics['maxDist'] = dists.max(axis=1)
    statistics['meanDist'] = dists.min(axis=1)

    X = pd.concat([X, statistics], axis=1)
    return X

# some heuristic settings
rfe_min_features = 12
rfe_step = 15
rfe_cv = 20
sss_n_splits = 20           
sss_test_size = 0.35
grid_search_cv = 20
noise_std = 0.0
r2_threshold = 0.185
random_seed = 234587

np.random.seed(random_seed)

# import data
train = pd.read_csv('../input/train.csv')
train_y = train['target']
train_X = train.drop(['id','target'], axis=1)

test = pd.read_csv('../input/test.csv')
test = test.drop(['id'], axis=1)

train_X = train_X.values
test = test.values

# scale using RobustScaler
data = RobustScaler().fit_transform(np.concatenate((train_X, test), axis=0))
train_X = data[:train_X.shape[0]]
test = data[train_X.shape[0]:]


train_X = with_statistics(pd.DataFrame(train_X)).values
test = with_statistics(pd.DataFrame(test)).values

# noise does not help with Logistic Regression
# train_X += np.random.normal(0, noise_std, train_X.shape)

# define roc_auc_metric robust to only one class in y_pred
def scoring_roc_auc(y, y_pred):
    try:
        return roc_auc_score(y, y_pred)
    except:
        return 0.5

robust_roc_auc = make_scorer(scoring_roc_auc)

# define model and its parameters
model = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced', C=0.31, penalty='l1')

param_grid = {
        'C'     : [0.2, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37],
        'tol'   : [0.0001, 0.00011, 0.00009]
    }

grid_search = GridSearchCV(model, param_grid=param_grid, verbose=0, n_jobs=-1, scoring=robust_roc_auc, cv=20)
grid_search.fit(train_X, train_y)

# define recursive elimination feature selector
feature_selector = RFECV(grid_search.best_estimator_, min_features_to_select=rfe_min_features, scoring=robust_roc_auc, step=rfe_step, verbose=0, cv=20, n_jobs=-1)

print("counter | val_mse  |  val_mae  |  val_roc  |  val_cos  |  val_dist  |  val_r2    | best_score | feature_count ")
print("-------------------------------------------------------------------------------------------------")

predictions = pd.DataFrame()
counter = 0
# split training data to build one model on each traing-data-subset
for train_index, val_index in StratifiedShuffleSplit(n_splits=sss_n_splits, test_size=sss_test_size, random_state=random_seed).split(train_X, train_y):
    X, val_X = train_X[train_index], train_X[val_index]
    y, val_y = train_y[train_index], train_y[val_index]

    # get the best features for this data set
    feature_selector.fit(X, y)

    # remove irrelevant features from X, val_X and test
    X_important_features        = feature_selector.transform(X)
    val_X_important_features    = feature_selector.transform(val_X)
    test_important_features     = feature_selector.transform(test)

    # run grid search to find the best Lasso parameters for this subset of training data and subset of features 
    grid_search = GridSearchCV(feature_selector.estimator_, param_grid=param_grid, verbose=0, n_jobs=-1, scoring=robust_roc_auc, cv=grid_search_cv)
    grid_search.fit(X_important_features, y)
    y_pred = grid_search.best_estimator_.predict_proba(X_important_features)[:,1]

    # score our fitted model on validation data
    val_y_pred = grid_search.best_estimator_.predict_proba(val_X_important_features)[:,1]
    val_mse = mean_squared_error(val_y, val_y_pred)
    val_mae = mean_absolute_error(val_y, val_y_pred)
    val_roc = roc_auc_score(val_y, val_y_pred)
    val_cos = cosine_similarity(val_y.values.reshape(1, -1), val_y_pred.reshape(1, -1))[0][0]
    val_dst = euclidean_distances(val_y.values.reshape(1, -1), val_y_pred.reshape(1, -1))[0][0]
    val_r2  = r2_score(val_y, val_y_pred)

    # if model did well on validation, save its prediction on test data, using only important features
    # r2_threshold (0.185) is a heuristic threshold for r2 error
    # you can use any other metric/metric combination that works for you
    if val_r2 > r2_threshold:
        message = '<-- OK'
        prediction = grid_search.best_estimator_.predict_proba(test_important_features)[:,1]
        predictions = pd.concat([predictions, pd.DataFrame(prediction)], axis=1)
    else:
        message = '<-- skipping'


    print("{:2}      | {:.4f}   |  {:.4f}   |  {:.4f}   |  {:.4f}   |  {:.4f}    |  {:.4f}    |  {:.4f}    |  {:3}         {}  ".format(counter, val_mse, val_mae, val_roc, val_cos, val_dst, val_r2, grid_search.best_score_, feature_selector.n_features_, message))
    
    counter += 1


print("-------------------------------------------------------------------------------------------------")
print("{}/{} models passed validation threshold and will be ensembled.".format(len(predictions.columns), sss_n_splits))

mean_pred = pd.DataFrame(predictions.mean(axis=1))
mean_pred.index += 250
mean_pred.columns = ['target']
mean_pred.to_csv('submission.csv', index_label='id', index=True)       

