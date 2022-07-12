# Note: Kaggle only runs Python 3, not Python 2

#code skeleton for feature engineering and hyperparameter search for xgboost
#now I give out the optimal parameters for passing the KS test

#do you know you can 'pip install xgboost'? 
#if you have problem pip install, go https://github.com/dmlc/xgboost and raise an issue.

import numpy as np
import pandas as pd
import xgboost as xgb
import sys

#import evaluation
exec(open("../input/evaluation.py").read())

from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV

import math

""" Implemented Scikit- style grid search to find optimal XGBoost params"""
""" Use this module to identify optimal hyperparameters for XGBoost"""

#have some feature engineering work for better rank
def add_features(df):
    #significance of flight distance
    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
    #Stepan Obraztsov's magic features
    df['NEW_FD_SUMP']=df['FlightDistance']/(df['p0_p']+df['p1_p']+df['p2_p'])
    df['NEW5_lt']=df['LifeTime']*(df['p0_IP']+df['p1_IP']+df['p2_IP'])/3
    df['p_track_Chi2Dof_MAX'] = df.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)
    #some more magic features
    df['p0p2_ip_ratio']=df['IP']/df['IP_p0p2']
    df['p1p2_ip_ratio']=df['IP']/df['IP_p1p2']
    df['DCA_MAX'] = df.loc[:, ['DOCAone', 'DOCAtwo', 'DOCAthree']].max(axis=1)
    df['iso_bdt_min'] = df.loc[:, ['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT']].min(axis=1)
    df['iso_min'] = df.loc[:, ['isolationa', 'isolationb', 'isolationc','isolationd', 'isolatione', 'isolationf']].min(axis=1)

    return df

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test = pd.read_csv("../input/test.csv")

print("Adding features to both training and testing")
train = add_features(train)
test = add_features(test)

print("Loading check agreement for KS test evaluation")
check_agreement = pd.read_csv('../input/check_agreement.csv')
check_correlation = pd.read_csv('../input/check_correlation.csv')
check_agreement = add_features(check_agreement)
check_correlation = add_features(check_correlation)
train_eval = train[train['min_ANNmuon'] > 0.4]

print("Eliminate SPDhits, which makes the agreement check fail")
filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal','SPDhits','p0_track_Chi2Dof','CDF1', 'CDF2', 'CDF3','isolationb', 'isolationc','p0_pt', 'p1_pt', 'p2_pt', 'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta','DOCAone', 'DOCAtwo', 'DOCAthree']
features = list(f for f in train.columns if f not in filter_out)
print("features:",features)

print("Train a XGBoost model")

xgb_model = xgb.XGBClassifier()

#when in doubt, use xgboost
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.15], #so called `eta` value
              'max_depth': [8],
              'min_child_weight': [3,11],
              'silent': [1],
              'subsample': [0.9],
              'colsample_bytree': [0.5],
              'n_estimators': [300], #number of trees
              'seed': [1337]}

#evaluate with roc_auc_truncated
def _score_func(estimator, X, y):
    pred_probs = estimator.predict_proba(X)[:, 1]
    return roc_auc_truncated(y, pred_probs)
    
#should evaluate by train_eval instead of the full dataset
clf = GridSearchCV(xgb_model, parameters, n_jobs=4, 
                   cv=StratifiedKFold(train_eval['signal'], n_folds=5, shuffle=True), 
                    scoring=_score_func,
                   verbose=2, refit=True)

clf.fit(train[features], train["signal"])

best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('Raw AUC score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

test_agreement_ = True
if test_agreement_:
    agreement_probs= (clf.predict_proba(check_agreement[features])[:,1])

    ks = compute_ks(
            agreement_probs[check_agreement['signal'].values == 0],
            agreement_probs[check_agreement['signal'].values == 1],
            check_agreement[check_agreement['signal'] == 0]['weight'].values,
            check_agreement[check_agreement['signal'] == 1]['weight'].values)
    print ('KS metric', ks, ks < 0.09)

    correlation_probs = clf.predict_proba(check_correlation[features])[:,1]
    print ('Checking correlation...')
    cvm = compute_cvm(correlation_probs, check_correlation['mass'])
    print ('CvM metric', cvm, cvm < 0.002)

    train_eval_probs = clf.predict_proba(train_eval[features])[:,1]
    print ('Calculating AUC...')
    AUC = roc_auc_truncated(train_eval['signal'], train_eval_probs)
    print ('AUC', AUC)

test_probs = clf.predict_proba(test[features])[:,1]
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("xgboost_best_parameter_submission.csv", index=False)