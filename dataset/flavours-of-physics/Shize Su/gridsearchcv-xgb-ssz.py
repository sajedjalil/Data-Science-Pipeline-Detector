"""

I have modified Ben Hammer's script to accomodate calculation of 
agreement, correlation and weighted roc score before submission.

If you see that its below required threshold then don't submit it.

original author: Ben Hammer.

modifications: Harshaneel Gokhale.

"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import *
from sklearn import ensemble, tree
import xgboost as xgb
import sys
sys.path.append('../input')
import evaluation
from sklearn.ensemble import GradientBoostingClassifier
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
    #df['NEW']=df['IP']*df['dira']
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
#features = list(train.columns[1:-5])

filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal','SPDhits','p0_track_Chi2Dof','CDF1', 'CDF2', 'CDF3','isolationb', 'isolationc','p0_pt', 'p1_pt', 'p2_pt', 'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta','DOCAone', 'DOCAtwo', 'DOCAthree']
#filter_out = ['id', 'min_ANNmuon','production','signal','SPDhits','CDF1', 'CDF2', 'CDF3','isolationb', 'isolationc','p0_pt', 'p1_pt', 'p2_pt', 'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta','DOCAone', 'DOCAtwo', 'DOCAthree']

features = list(f for f in train.columns if f not in filter_out)

train_eval = train[train['min_ANNmuon'] > 0.4]

print("features:",features)
#train[features] = train[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#test[features] = test[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#check_agreement[features] = check_agreement[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#check_correlation[features] = check_correlation[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

print("Train a Random Forest model")
#rf1 = RandomForestClassifier(n_estimators=500, n_jobs=-1, criterion="entropy", max_depth=6, random_state=1)
rf1 = RandomForestClassifier(n_estimators=600, n_jobs=-1, criterion="entropy", max_depth=10, max_features=6, min_samples_leaf=2, random_state=1)
rf1.fit(train[features], train["signal"])
#rf = ensemble.AdaBoostClassifier(n_estimators=50, learning_rate=0.098643,base_estimator=rf1)
#uniform_features  = ["mass"]
print("Train a UGradientBoostingClassifier")
loss = BinFlatnessLossFunction(['mass'], n_bins=15, uniform_label=0)
#loss = KnnFlatnessLossFunction(uniform_features, uniform_label=0)
rf = UGradientBoostingClassifier(loss=loss, n_estimators=300,  
                                  max_depth=6,
                                  learning_rate=0.15, train_features=features, subsample=0.7, random_state=369)
rf.fit(train[features + ['mass']], train['signal'])
#loss_funct=LogLossFunction()
#rf=UGradientBoostingClassifier(loss=loss_funct,n_estimators=200, random_state=3,learning_rate=0.2,subsample=0.7)
#rf.fit(train[features],train["signal"])

print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "learning_rate": 0.2,
          "max_depth": 6,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
          
num_trees=450

#clf = GridSearchCV(xgb_model, params, n_jobs=4, 
 #                  cv=StratifiedKFold(train['signal'], n_folds=5, shuffle=True), 
 #                  verbose=2, refit=True)

#clf.fit(train[features], train["signal"])

#best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
#print('Raw AUC score:', score)
#for param_name in sorted(best_parameters.keys()):
#    print("%s: %r" % (param_name, best_parameters[param_name]))


gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)

#agreement_probs= 0.5*rf.predict_proba(check_agreement[features])[:,1] + 0.5* gbm.predict(xgb.DMatrix(check_agreement[features]))
agreement_probs= 0.4*rf.predict_proba(check_agreement[features])[:,1] + 0.4* gbm.predict(xgb.DMatrix(check_agreement[features])) + 0.2*rf1.predict_proba(check_agreement[features])[:,1]

print('Checking agreement...')
ks = evaluation.compute_ks(
	agreement_probs[check_agreement['signal'].values == 0],
	agreement_probs[check_agreement['signal'].values == 1],
	check_agreement[check_agreement['signal'] == 0]['weight'].values,
	check_agreement[check_agreement['signal'] == 1]['weight'].values)
print ('KS metric', ks, ks < 0.09)

#correlation_probs = 0.5*rf.predict_proba(check_correlation[features])[:,1] + 0.5*gbm.predict(xgb.DMatrix(check_correlation[features]))
correlation_probs = 0.4*rf.predict_proba(check_correlation[features])[:,1] + 0.4*gbm.predict(xgb.DMatrix(check_correlation[features])) + 0.2*rf1.predict_proba(check_correlation[features])[:,1]

print ('Checking correlation...')
cvm = evaluation.compute_cvm(correlation_probs, check_correlation['mass'])
print ('CvM metric', cvm, cvm < 0.002)

#train_eval_probs = 0.5*rf.predict_proba(train_eval[features])[:,1] + 0.5*gbm.predict(xgb.DMatrix(train_eval[features]))
train_eval_probs = 0.4*rf.predict_proba(train_eval[features])[:,1] + 0.4*gbm.predict(xgb.DMatrix(train_eval[features])) + 0.2*rf1.predict_proba(train_eval[features])[:,1]

print ('Calculating AUC...')
AUC = evaluation.roc_auc_truncated(train_eval['signal'], train_eval_probs)
print ('AUC', AUC)

train_eval_probs1 = rf.predict_proba(train_eval[features])[:,1]
AUC1 = evaluation.roc_auc_truncated(train_eval['signal'], train_eval_probs1)
print ('AUC UB', AUC1)

train_eval_probs2 = gbm.predict(xgb.DMatrix(train_eval[features]))
AUC2 = evaluation.roc_auc_truncated(train_eval['signal'], train_eval_probs2)
print ('AUC XGB', AUC2)

train_eval_probs3 = rf1.predict_proba(train_eval[features])[:,1]
AUC3 = evaluation.roc_auc_truncated(train_eval['signal'], train_eval_probs3)
print ('AUC UB', AUC3)

print("Make predictions on the test set")
test_probs = 0.4*rf.predict_proba(test[features])[:,1] + 0.4*gbm.predict(xgb.DMatrix(test[features])) + 0.2*rf.predict_proba(test[features])[:,1] 
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("ub_rf_xgboost_submission.csv", index=False)
