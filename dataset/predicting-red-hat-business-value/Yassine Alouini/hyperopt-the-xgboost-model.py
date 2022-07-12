# Run an XGBoost model with hyperparmaters that are optimized using hyperopt
# The output of the script are the best hyperparmaters
# The optimization part using hyperopt is partly inspired from the following script: 
# https://github.com/bamine/Kaggle-stuff/blob/master/otto/hyperopt_xgboost.py


# Data wrangling

import pandas as pd

# Scientific 

import numpy as np


# Machine learning

import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

# Hyperparameters tuning

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# Some constants

SEED = 314159265
VALID_SIZE = 0.2
TARGET = 'outcome'

#-------------------------------------------------#

# Utility functions

def intersect(l_1, l_2):
    return list(set(l_1) & set(l_2))


def get_features(train, test):
    intersecting_features = intersect(train.columns, test.columns)
    intersecting_features.remove('people_id')
    intersecting_features.remove('activity_id')
    return sorted(intersecting_features)

#-------------------------------------------------#

# Scoring and optimization functions


def score(params):
    print("Training with params: ")
    print(params)
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = xgb.DMatrix(train_features, label=y_train)
    dvalid = xgb.DMatrix(valid_features, label=y_valid)
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm_model = xgb.train(params, dtrain, num_round,
                          evals=watchlist,
                          verbose_eval=True)
    predictions = gbm_model.predict(dvalid,
                                    ntree_limit=gbm_model.best_iteration + 1)
    score = roc_auc_score(y_valid, predictions)
    # TODO: Add the importance for the selected features
    print("\tScore {0}\n\n".format(score))
    # The score function should return the loss (1-score)
    # since the optimize function looks for the minimum
    loss = 1 - score
    return {'loss': loss, 'status': STATUS_OK}


def optimize(
             #trials, 
             random_state=SEED):
    """
    This is the optimization function that given a space (space here) of 
    hyperparameters and a scoring function (score here), finds the best hyperparameters.
    """
    # To learn more about XGBoost parameters, head to this page: 
    # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
        # A problem with max_depth casted to float instead of int with
        # the hp.quniform method.
        'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        # Increase this number if you have more cores. Otherwise, remove it and it will default 
        # to the maxium number. 
        'nthread': 4,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'silent': 1,
        'seed': random_state
    }
    # Use the fmin function from Hyperopt to find the best hyperparameters
    best = fmin(score, space, algo=tpe.suggest, 
                # trials=trials, 
                max_evals=250)
    return best

#-------------------------------------------------#


# Load processed data

# You could use the following script to generate a well-processed train and test data sets:
# https://www.kaggle.com/yassinealouini/predicting-red-hat-business-value/features-processing
# I have only used the .head() of the data sets since the process takes a long time to run.
# I have also put the act_train and act_test data sets since I don't have the processed data sets 
# loaded. 

train_df = pd.read_csv('../input/act_train.csv').head(100)
test_df = pd.read_csv('../input/act_test.csv').head(100)

FEATURES = get_features(train_df, test_df)
print(FEATURES)


#-------------------------------------------------#



# Extract the train and valid (used for validation) dataframes from the train_df

train, valid = train_test_split(train_df, test_size=VALID_SIZE,
                                random_state=SEED)
train_features = train[FEATURES]
valid_features = valid[FEATURES]
y_train = train[TARGET]
y_valid = valid[TARGET]

print('The training set is of length: ', len(train.index))
print('The validation set is of length: ', len(valid.index))

#-------------------------------------------------#

# Run the optimization

# Trials object where the history of search will be stored
# For the time being, there is a bug with the following version of hyperopt.
# You can read the error messag on the log file.
# For the curious, you can read more about it here: https://github.com/hyperopt/hyperopt/issues/234
# => So I am commenting it.
# trials = Trials()

best_hyperparams = optimize(
                            #trials
                            )
print("The best hyperparameters are: ", "\n")
print(best_hyperparams)
