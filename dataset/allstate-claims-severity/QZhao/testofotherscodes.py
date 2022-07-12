# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import datetime
pd.set_option('display.max_columns', None)
#pd.set_option('display.expand_frame_repr', False)
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from operator import itemgetter
import time
from sklearn import preprocessing
from collections import Counter
from scipy.stats import skew
from sklearn.feature_selection import SelectKBest

############################################################################
# Function to Write result in csv file to submit 
###########################################################################

def write_to_csv(output,score):
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    prediction_file_object = csv.writer(f)
    prediction_file_object.writerow(["id","loss"])  # don't forget the headers

    for i in range(len(test)):
        prediction_file_object.writerow([test["id"][test.index[i]], (output[i])])


############################################################################
# Function to process features 
###########################################################################
def get_features(train, test):
    trainval = list(train.columns.values) # list train features
    testval = list(test.columns.values) # list test features
    output = list(set(trainval) & set(testval)) # check wich features are in common (remove the outcome column)
    output.remove('id') # remove non-usefull id column

    return output
    
def process_features(train,test):
    
    table = pd.concat([train, test]).reset_index() # put all data in one dataframe

    table.loss=np.log(table.loss) # using log as feature is very skewed

    print ("Handling categorical features...")

    categorical_features=table.select_dtypes(include=["object"]).columns.values
    for feature in categorical_features:
        table[feature] = pd.factorize(table[feature], sort=True)[0]

    ## dummy encoding -> end up with 1100+ features, not a good solution
    #dummies=pd.get_dummies(table[categorical_features])
    #table = pd.concat([table,dummies],axis=1)
    #table.drop(categorical_features,axis=1,inplace=True)

    
### get the original tables back
    train=table[table.index<len(train)]
    test=table[table.index>=len(train)].reset_index()
    test.drop(["loss","index"],axis=1,inplace=True)
    train.drop("index",axis=1,inplace=True)
    print ("Getting features...")
    features = get_features(train,test)
    
    return train,test,features

def train_and_test_linear(train,test,features,target='SalePrice'): # simple xgboost
    subsample = 0.8
    colsample_bytree = 0.8
    num_boost_round = 1000 #115 originally 
    early_stopping_rounds = 50
    test_size = 0.2 # 0.1 originally
    
    start_time = time.time()
   
    # start the training

    params = {
        "objective": "reg:linear",
        "booster" : "gblinear", #"gbtree",# default
        "eval_metric": "rmse",
        "subsample": subsample, # collect 80% of the data only to prevent overfitting
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": 0,
    }

    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=0) # randomly split into 90% test and 10% CV -> still has the outcome at this point
    y_train = np.log(X_train[target]) # define y as the outcome column, apply log to have same error as the leaderboard
    y_valid = np.log(X_valid[target])
    dtrain = xgb.DMatrix(X_train[features], y_train) # DMatrix are matrix for xgboost
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')] # list of things to evaluate and print
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True) # find the best score
    score = gbm.best_score #roc_auc_score(X_valid[target].values, check)
    print('Last error value: {:.6f}'.format(score))

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]))

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))

    return test_prediction, score    

def train_and_test_tree(train,test,features,target='SalePrice'): # simple xgboost
    eta_list = [0.1,0.2] # list of parameters to try
    max_depth_list = [4,6,8] # list of parameters to try
    subsample = 0.8
    colsample_bytree = 0.8
    
    num_boost_round = 400 
    early_stopping_rounds = 10
    test_size = 0.2 
    
    start_time = time.time()
   
    # start the training
    array_score=np.ndarray((len(eta_list)*len(max_depth_list),3)) # store score values
    i=0
    for eta,max_depth in list(itertools.product(eta_list, max_depth_list)): # Loop over parameters to find the better set
        print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
        params = {
            "objective": "reg:linear",
            "booster" : "gbtree", 
            "eval_metric": "rmse", # this is the metric for the leardboard
            "eta": eta, # shrinking parameters to prevent overfitting
            "tree_method": 'exact',
            "max_depth": max_depth,
            "subsample": subsample, # collect 80% of the data only to prevent overfitting
            "colsample_bytree": colsample_bytree,
            "silent": 1,
            "seed": 0,
        }
    
        X_train, X_valid = train_test_split(train, test_size=test_size, random_state=0) # randomly split into 90% test and 10% CV -> still has the outcome at this point
        y_train = (X_train[target]) # define y as the outcome column
        y_valid = (X_valid[target])
        dtrain = xgb.DMatrix(X_train[features], y_train) # DMatrix are matrix for xgboost
        dvalid = xgb.DMatrix(X_valid[features], y_valid)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')] # list of things to evaluate and print
        gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True) # find the best score

        print("Validating...")
        score = gbm.best_score 
        print('Last error value: {:.6f}'.format(score))
        array_score[i][0]=eta
        array_score[i][1]=max_depth
        array_score[i][2]=score
        i+=1
    df_score=pd.DataFrame(array_score,columns=['eta','max_depth','SalePrice'])
    print("df_score : \n", df_score)
    #create_feature_map(features)
    importance = gbm.get_fscore()
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    print('Importance array: ', importance)
    np.save("features_importance",importance) # save feature importance for latter use
    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_ntree_limit) # only predict with the last set of parameters

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))

    return test_prediction, score 

def train_and_test_Kfold(train,test,features,target='loss'): # add Kfold
    eta_list = [0.08] # list of parameters to try
    max_depth_list = [4]
    min_child_weight_list=[5]
    gamma_list=[0]
    subsample_list = [0.85] # No subsampling, as we already use Kfold latter and we don't have that much data
    colsample_bytree_list = [0.9]
    
    
    num_boost_round = 100 # for small eta, increase this one
    early_stopping_rounds = 20
    n_folds=3
    start_time = time.time()
   

    # start the training
    array_score=np.ndarray((len(eta_list)*len(max_depth_list)*len(min_child_weight_list)*len(gamma_list)*len(subsample_list)*len(colsample_bytree_list),8)) # store score values
    i=0
    for eta,max_depth,min_child_weight,gamma,subsample,colsample_bytree in list(itertools.product(eta_list, max_depth_list,min_child_weight_list,gamma_list,subsample_list,colsample_bytree_list)): # Loop over parameters to find the better set
        print('XGBoost params. ETA: {}, MAX_DEPTH: {}'.format(eta, max_depth))
        params = {
            "objective": "reg:linear",
            "booster" : "gbtree",
            "eval_metric": "mae",
            "eta": eta, # shrinking parameters to prevent overfitting
            "tree_method": 'exact',
            "max_depth": max_depth,
            "min_child_weight" : min_child_weight,
            "subsample": subsample, # collect 80% of the data only to prevent overfitting
            "colsample_bytree": colsample_bytree, #colsample_bytree,
            "silent": 1,
            "gamma":gamma,
            "seed": 118,
        }
        kf = KFold(len(train), n_folds=n_folds,shuffle=True)
        test_prediction=np.ndarray((n_folds,len(test)))
        fold=0
        fold_score=[]
        for train_index, cv_index in kf:
            X_train, X_valid    = train[features].as_matrix()[train_index], train[features].as_matrix()[cv_index]
            y_train, y_valid    = (train[target].as_matrix()[train_index]), (train[target].as_matrix()[cv_index])

            dtrain = xgb.DMatrix(X_train, y_train) # DMatrix are matrix for xgboost
            dvalid = xgb.DMatrix(X_valid, y_valid)
            dtrain_exp = xgb.DMatrix(X_train, np.exp(y_train))
            dvalid_exp = xgb.DMatrix(X_valid, np.exp(y_valid))
            
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')] # list of things to evaluate and print
            gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True) # find the best score

            print("Validating...")
            check = gbm.predict(xgb.DMatrix(X_valid)) # get the best score
            score = gbm.best_score
            print('Check last score value: {:.6f}'.format(score))
            fold_score.append(score)
            importance = gbm.get_fscore()
            sorted_importance=(dict((int(key.lstrip("f")), value) for (key, value) in importance.items()))
            feature_importance=dict(zip([features[j] for j in sorted_importance.keys()],sorted_importance.values()))
            if fold==0:
                feature_sum=Counter(feature_importance)
            else:
                feature_sum+=Counter(feature_importance)
            print("Predict test set...")
            prediction=gbm.predict(xgb.DMatrix(test[features].as_matrix()))
            test_prediction[fold]=prediction
            fold = fold + 1
        mean_score=np.mean(fold_score)
        importance = sorted(feature_sum.items(), key=itemgetter(1), reverse=True)
        #np.save(("features_importance_depth_{}").format(max_depth),importance)
        print('Total Importance array  :\n {}'.format(importance))
        array_score[i][0]=eta
        array_score[i][1]=max_depth
        array_score[i][2]=min_child_weight
        array_score[i][3]=gamma
        array_score[i][4]=subsample
        array_score[i][5]=colsample_bytree
        array_score[i][6]=mean_score
        array_score[i][7]=np.std(fold_score)
        i+=1
    final_prediction=test_prediction.mean(axis=0)
    df_score=pd.DataFrame(array_score,columns=['eta','max_depth','min_child_weight','gamma','subsample','colsample_bytree','mean_score','std_score'])
    print ("df_score : \n {}".format(df_score))# get the complete array of scores to choose the right parameters

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))

    return final_prediction, mean_score 


############################################################################
# Main code
###########################################################################

#num_features = None # Choose how many features you want to use. None = all

train = pd.read_csv("../input/train.csv") # read train data
test = pd.read_csv("../input/test.csv") # read test data
#train = pd.read_csv("train.csv") # read train data
#test = pd.read_csv("test.csv") # read test data
train,test,features = process_features(train,test)

test_prediction,score = train_and_test_Kfold(train,test,features[:]) 
write_to_csv(np.exp(test_prediction),score) 

## Note that the obtained score is the differences of the log(loss), which is different (in value) from the LB score, but it is still usefull to see if you improved 
