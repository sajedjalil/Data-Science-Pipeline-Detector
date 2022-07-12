"""

I have modified Ben Hammer's script to accomodate calculation of 
agreement, correlation and weighted roc score before submission.

If you see that its below required threshold then don't submit it.

original author: Ben Hammer.

modifications: Harshaneel Gokhale.
last modifications: Justfor

"""
import numpy as np
import pandas as pd
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import *
from sklearn import ensemble, tree
import xgboost as xgb
import sys
sys.path.append('../input')
import evaluation
#from sklearn.ensemble import GradientBoostingClassifier
from hep_ml.uboost import uBoostClassifier
from hep_ml.gradientboosting import UGradientBoostingClassifier,LogLossFunction
from hep_ml.losses import BinFlatnessLossFunction, KnnFlatnessLossFunction

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")
check_agreement = pd.read_csv('../input/check_agreement.csv')
check_correlation = pd.read_csv('../input/check_correlation.csv')

#have some feature engineering work for better rank
def add_features(df):
    #significance of flight distance
    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
    df['NEW_IP_dira']=df['IP']*df['dira']
    df['NEW_FD_SUMP']=df['FlightDistance']/(df['p0_p']+df['p1_p']+df['p2_p'])
    df['NEW5_lt']=df['LifeTime']*(df['p0_IP']+df['p1_IP']+df['p2_IP'])/3
    df['p_track_Chi2Dof_MAX'] = df.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)
    return df
print("Adding features to both training and testing")
train = add_features(train)
test = add_features(test)

check_agreement = add_features(check_agreement)
check_correlation = add_features(check_correlation)

print("Eliminate SPDhits, which makes the agreement check fail")
filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal','SPDhits','p0_track_Chi2Dof','CDF1', 'CDF2', 'CDF3','isolationb', 'isolationc','p0_pt', 'p1_pt', 'p2_pt', 'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta','DOCAone', 'DOCAtwo', 'DOCAthree']

features = list(f for f in train.columns if f not in filter_out)

train_eval = train[train['min_ANNmuon'] > 0.4]

print("features:",features)
print("Train a UGradientBoostingClassifier")
loss = BinFlatnessLossFunction(['mass'], n_bins=20, uniform_label=0)
rf = UGradientBoostingClassifier(loss=loss, n_estimators=425,  
                                  max_depth=7,
                                  learning_rate=0.17, train_features=features, subsample=0.8, random_state=369)
rf.fit(train[features + ['mass']], train['signal'])



print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "learning_rate": 0.2,
          "max_depth": 8,
          'gamma': 0.01,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.8,
          'nthread': 4,
          "seed": 1}
          
num_trees=550

gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)

agreement_probs= 0.5*rf.predict_proba(check_agreement[features])[:,1] + 0.5* gbm.predict(xgb.DMatrix(check_agreement[features]))
print('Checking agreement...')
ks = evaluation.compute_ks(
	agreement_probs[check_agreement['signal'].values == 0],
	agreement_probs[check_agreement['signal'].values == 1],
	check_agreement[check_agreement['signal'] == 0]['weight'].values,
	check_agreement[check_agreement['signal'] == 1]['weight'].values)
print ('KS metric 0.5:', ks, ks < 0.09)

'''
correlation_probs = 0.5*rf.predict_proba(check_correlation[features])[:,1] + 0.5*gbm.predict(xgb.DMatrix(check_correlation[features]))
print ('Checking correlation...')
cvm = evaluation.compute_cvm(correlation_probs, check_correlation['mass'])
print ('CvM metric', cvm, cvm < 0.002)
'''

print ('Calculating AUC...')
train_eval_probs_rf = rf.predict_proba(train_eval[features])[:,1]
AUC = evaluation.roc_auc_truncated(train_eval['signal'], train_eval_probs_rf)
print ('AUC UB ', AUC)

train_eval_probs_gbm = gbm.predict(xgb.DMatrix(train_eval[features]))
AUC = evaluation.roc_auc_truncated(train_eval['signal'], train_eval_probs_gbm)
print ('AUC XGB', AUC)

train_eval_probs = 0.5*train_eval_probs_rf + 0.5*train_eval_probs_gbm
AUC = evaluation.roc_auc_truncated(train_eval['signal'], train_eval_probs)
print ('AUC 0.5', AUC)

train_eval_probs = 0.6*train_eval_probs_rf + 0.4*train_eval_probs_gbm
AUC = evaluation.roc_auc_truncated(train_eval['signal'], train_eval_probs)
print ('AUC 0.6', AUC)

print("Make predictions on the test set")
rfpred = rf.predict_proba(test[features])[:,1]
gbmpred = gbm.predict(xgb.DMatrix(test[features]))

print("Make predictions on the test set xgb - ub")
test_probs = 0.5* rfpred + 0.5 * gbmpred
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("ub_xgboost_submission.csv", index=False)

print("Make predictions on the test set xgb - ub")
test_probs = 0.6* rfpred + 0.4 * gbmpred
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("ub_xgboost_6_4_submission.csv", index=False)
                