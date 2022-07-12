# -*- coding: utf-8 -*-
"""
@author: Michael
Credits: https://towardsdatascience.com/an-introductory-example-of-bayesian-optimization-in-python-with-hyperopt-aae40fff4ff0
Notes
v19 was the result of fine-tuning done on earlier versions so the search space was quite specific. 
v21 reverts to a larger search space. Also change cv seed in case of fitting to the previous one
"""

# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# Modeling
import lightgbm as lgb
import xgboost as xgb

# Evaluation of the model
from sklearn.model_selection import KFold
import random
import csv as csv
import datetime

MAX_EVALS = 30
N_FOLDS = 5

# Create the dataset
print("Loading")
train = pd.read_csv('../input/train.csv')

print("Processing")
train_labels = train['target']
train = train.drop(["ID_code"], axis = 1)
train = train.drop(["target"], axis = 1)
print("train head")
print(train.head())

col_mask = list(train)

train = train[col_mask]
#print(list(train))

# Convert to numpy array for splitting in cross validation
features = np.array(train)
test_features = np.array(train_labels)
labels = train_labels[:]

print('Dataset shape: ', features.shape)
print('Labels shape: ', test_features.shape)

import matplotlib.pyplot as plt
import seaborn as sns

# Model with default hyperparameters
model = xgb.XGBClassifier()
model

from sklearn.metrics import roc_auc_score
from timeit import default_timer as timer

print("Setting param grid")
# Hyperparameter grid
param_grid = {
    'tree_method': ['gpu_hist'],                    
    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'colsample_bytree': list(np.linspace(0.01, 1, 199)),
    'min_child_weight': list(np.linspace(0, 2000, 2001)),
    'max_delta_step' : list(np.linspace(1, 12, 12)),
    'reg_lambda': list(np.linspace(0, 4, 4001)),
    'reg_alpha': list(np.linspace(0, 4, 4001))
}

# Subsampling (only applicable with 'goss')
subsample_dist = list(np.linspace(0.25, 1, 151))

#establish xgb.cv parameters
nbr = 1000000
esr = 200
xgbcv_seed = 725
eta = 0.05

# Randomly sample parameters for gbm
params = {key: random.sample(value, 1)[0] for key, value in param_grid.items()}
params['subsample'] = random.sample(subsample_dist, 1)[0]
params['learning_rate'] = eta
params['verbose'] = -1
params['silent'] = True
print(params)

# Create xgb dataset
print("Create xgboost dataset")
train_set = xgb.DMatrix(features, train_labels, feature_names=train.columns) #lgb.Dataset(features, label = labels)

#Get a benchmark score
print("Get a benchmark untuned score") #it was more untuned in earlier versions, the search space has narrowed through the versions
r = xgb.cv(params, train_set, num_boost_round = nbr, nfold = N_FOLDS, seed=xgbcv_seed, stratified=True, metrics='auc', 
           early_stopping_rounds = esr, verbose_eval=False)

# Highest score
r_best = np.max(r['test-auc-mean'])

# Standard deviation of best score
r_best_std = r['test-auc-std'][r['test-auc-mean'].idxmax()]

print('The maximium ROC AUC on the validation set was {:.5f} with std of {:.5f}.'.format(r_best, r_best_std))
print('The ideal number of iterations was {}.'.format(r['test-auc-mean'].idxmax() + 1))

# Dataframe to hold cv results
random_results = pd.DataFrame(columns = ['loss', 'params', 'iteration', 'estimators', 'time'],
                       index = list(range(MAX_EVALS)))

def random_objective(params, iteration, n_folds = N_FOLDS):
    """Random search objective function. Takes in hyperparameters
       and returns a list of results to be saved."""

    start = timer()
    
    # Perform n_folds cross validation
    cv_results = xgb.cv(params, train_set, num_boost_round = nbr, nfold = N_FOLDS, seed=xgbcv_seed, stratified=True, metrics='auc', 
                        early_stopping_rounds = esr, verbose_eval=False)
    end = timer()
    best_score = np.max(cv_results['test-auc-mean'])
    print(np.round(best_score,5))
    
    # Loss must be minimized
    loss = 1 - best_score
    
    # Boosting rounds that returned the highest cv score
    n_estimators = int(cv_results['test-auc-mean'].idxmax() + 1)
    
    # Return list of results
    return [loss, params, iteration, n_estimators, end - start]

random.seed(50)

# Iterate through the specified number of evaluations
print("Iterate to tune hyperparameters")
for i in range(MAX_EVALS):
    
    print("Running Iteration: " + str(i))
    
    # Randomly sample parameters for gbm
    params = {key: random.sample(value, 1)[0] for key, value in param_grid.items()}
    params['subsample'] = random.sample(subsample_dist, 1)[0]
    params['learning_rate'] = eta

    print(params)
           
    results_list = random_objective(params, i)
    
    # Add results to next row in dataframe
    random_results.loc[i, :] = results_list
    
# Sort results by best validation score
random_results.sort_values('loss', ascending = True, inplace = True)
random_results.reset_index(inplace = True, drop = True)
random_results.head()

#Display the Best Results
print("Best Parameters Found")
random_results.loc[0, 'params']

best_auc = 1 - random_results.loc[0, 'loss']

print('The best cv score from random search scores {:.5f}.'.format(best_auc))
print('This was achieved on search iteration {}.'.format(random_results.loc[0, 'iteration']))
print('With these parameters: ' + (str(random_results.loc[0, 'params'])))
print('After This Many Rounds of Training: ' + (str(random_results.loc[0, 'estimators'])))
print('Time taken to run best iteration (seconds): {:.0f}.'.format(random_results.loc[0, 'time']))


# Find the best parameters and number of estimators
best_random_params = random_results.loc[0, 'params'].copy()
best_random_estimators = int(random_results.loc[0, 'estimators'])


av_log_file = open("parameter_tuning.csv", "w", newline="")
open_file_object = csv.writer(av_log_file)
open_file_object.writerow(['The best cv score from random search scores {:.5f}.'.format(best_auc)])                            
open_file_object.writerow(['This was achieved using {} search iterations.'.format(random_results.loc[0, 'iteration'])])
open_file_object.writerow(['With these parameters: ' + (str(random_results.loc[0, 'params']))])
open_file_object.writerow(['After This Many Rounds of Training: ' + (str(random_results.loc[0, 'estimators']))])
open_file_object.writerow(['Time taken to run best iteration (seconds): {:.0f}.'.format(random_results.loc[0, 'time'])])

av_log_file.close()
