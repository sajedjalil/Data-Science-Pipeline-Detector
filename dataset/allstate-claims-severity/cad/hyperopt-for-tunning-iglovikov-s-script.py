__author__ = 'Vladimir Iglovikov' #Edited by Carlos Dutra

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold
from hyperopt import fmin, tpe, hp, Trials

from datetime import datetime

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
test['loss'] = np.nan
joined = pd.concat([train, test])

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))
    
#Parameters:
num_round = 4000 #Number of rounds
num_folds = 10 #Number of folds
num_trials = 50 #Number of trials

i_trial = 0

def objective_funcion(params):
    global i_trial
    global num_trials
    global num_folds
    i_trial = i_trial + 1
    
    print "Running for trial:", i_trial, "/", num_trials
    print "Running for parameters:"
    print params
       
    #Cross-Fold
    kf = KFold(188318, n_folds=num_folds ,shuffle=True)
    mae = []
    k = 0
    for train_index, cv_index in kf:
        k = k + 1
        now = datetime.now()
        print str(now.strftime("%Y-%m-%d-%H-%M"))
        print "Running for fold:", k, "/", num_folds
        train_fold = X.as_matrix()[train_index]
        train_cv = X.as_matrix()[cv_index]
        logloss_fold = (y.as_matrix()[train_index])
        logloss_cv = (y.as_matrix()[cv_index])
                  
        d_train_fold = xgb.DMatrix(train_fold, label = logloss_fold)
        d_train_cv = xgb.DMatrix(train_cv, label = logloss_cv)
         
        watchlist  = [(d_train_cv,'eval'), (d_train_fold,'train')]
        clf = xgb.train(params, d_train_fold,num_round,watchlist, 
                        verbose_eval = False,
                        early_stopping_rounds=30)
        
        #Avalida performance do modelo
        logloss_pred = clf.predict(d_train_cv)
        
        mae.append(mean_absolute_error(np.exp(logloss_cv)-200,
                                       np.exp(logloss_pred)-200))
        print "MAE:", np.average(mae) 
        #if np.average(mae) > 1115:
        #    break
    
    print "Trial retruning MAE:", np.average(mae)
    return(np.average(mae))

if __name__ == '__main__':
    for column in list(train.select_dtypes(include=['object']).columns):
        if train[column].nunique() != test[column].nunique():
            set_train = set(train[column].unique())
            set_test = set(test[column].unique())
            remove_train = set_train - set_test
            remove_test = set_test - set_train

            remove = remove_train.union(remove_test)
            def filter_cat(x):
                if x in remove:
                    return np.nan
                return x

            joined[column] = joined[column].apply(lambda x: filter_cat(x), 1)
            
        joined[column] = pd.factorize(joined[column].values, sort=True)[0]

    train = joined[joined['loss'].notnull()]
    test = joined[joined['loss'].isnull()]

    shift = 200
    y = np.log(train['loss'] + shift)
    ids = test['id']
    X = train.drop(['loss', 'id'], 1)
    X_test = test.drop(['loss', 'id'], 1)
    
    RANDOM_STATE = 2016
         
    #Search space of the hyperparameters
    params = {}
    params['booster'] = 'gbtree'
    params['objective'] = "reg:linear"
    params['eval_metric'] = 'mae'
    params['eta'] = hp.uniform('eta',0.004, 0.006)
    params['gamma'] = hp.uniform('gamma', 0.9, 1.2)
    params['alpha'] = hp.uniform('alpha', 0.9, 1.2)
    params['min_child_weight'] = hp.uniform('min_child_weight', 1, 1.5)
    params['colsample_bytree'] = hp.uniform('colsample_bytree', 0.6, 0.7)
    params['subsample'] = hp.uniform('subsample', 0.6, 0.7)
    params['max_depth'] = 12
    params['max_delta_step'] = 0
    params['random_state'] = 1000
    
    #Create Trials object to save the trials
    trials = Trials()
    best_params = fmin(fn=objective_funcion,
                space=params,
                algo=tpe.suggest,
                max_evals=num_trials,
                trials=trials)
    
    print best_params
        
    xgtrain = xgb.DMatrix(X, label=y)
    clf = xgb.train(best_params, xgtrain,num_round,
                    verbose_eval = False)
    xgtest = xgb.DMatrix(X_test)
    prediction = np.exp(clf.predict(xgtest)) - shift

    submission = pd.DataFrame()
    submission['loss'] = prediction
    submission['id'] = ids
    submission.to_csv('sub_v.csv', index=False)