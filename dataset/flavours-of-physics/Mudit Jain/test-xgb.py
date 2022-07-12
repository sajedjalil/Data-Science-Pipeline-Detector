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
#filter_out = ['id', 'min_ANNmuon','production','signal','SPDhits','CDF1', 'CDF2', 'CDF3','isolationb', 'isolationc','p0_pt', 'p1_pt', 'p2_pt', 'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta','DOCAone', 'DOCAtwo', 'DOCAthree']

features = list(f for f in train.columns if f not in filter_out)

train_eval = train[train['min_ANNmuon'] > 0.4]

print("features:",features)
#train[features] = train[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#test[features] = test[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#check_agreement[features] = check_agreement[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#check_correlation[features] = check_correlation[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

#loss_funct=LogLossFunction()
#rf=UGradientBoostingClassifier(loss=loss_funct,n_estimators=200, random_state=3,learning_rate=0.2,subsample=0.7)
#rf.fit(train[features],train["signal"])

print("Train a XGBoost model")
#params = {"objective": "binary:logistic",
#          "learning_rate": 0.2,
#          "max_depth": 6,
#          "min_child_weight": 3,
#          "silent": 1,
#          "subsample": 0.7,
#          "colsample_bytree": 0.7,
#          'nthread': 4,
#          "seed": 1}
#          
#num_trees=450

params = {"objective": "binary:logistic",
          "learning_rate": 0.003,
          "max_depth": 8,
          'gamma': 0.01, # 0.005
          "min_child_weight": 5,
          "silent": 1,
          "subsample": 0.85,
          "colsample_bytree": 0.7,
          'nthread': 4,
          "seed": 46}

num_trees=3333

# xgb_model = xgb.XGBClassifier()

#clf = GridSearchCV(xgb_model, params, n_jobs=4, 
 #                  cv=StratifiedKFold(train['signal'], n_folds=5, shuffle=True), 
 #                  verbose=2, refit=True)

#clf.fit(train[features], train["signal"])

#best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
#print('Raw AUC score:', score)
#for param_name in sorted(best_parameters.keys()):
#    print("%s: %r" % (param_name, best_parameters[param_name]))

gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)


agreement_probs= gbm.predict(xgb.DMatrix(check_agreement[features]))
#agreement_probs= 0.4*rf.predict_proba(check_agreement[features])[:,1] + 0.4* gbm.predict(xgb.DMatrix(check_agreement[features])) + 0.2*rf1.predict_proba(check_agreement[features])[:,1]

print('Checking agreement...')
ks = evaluation.compute_ks(
	agreement_probs[check_agreement['signal'].values == 0],
	agreement_probs[check_agreement['signal'].values == 1],
	check_agreement[check_agreement['signal'] == 0]['weight'].values,
	check_agreement[check_agreement['signal'] == 1]['weight'].values)
print ('KS metric 0.5:', ks, ks < 0.09)

correlation_probs = gbm.predict(xgb.DMatrix(check_correlation[features]))
#correlation_probs = 0.4*rf.predict_proba(check_correlation[features])[:,1] + 0.4*gbm.predict(xgb.DMatrix(check_correlation[features])) + 0.2*rf1.predict_proba(check_correlation[features])[:,1]

print ('Checking correlation...')
cvm = evaluation.compute_cvm(correlation_probs, check_correlation['mass'])
print ('CvM metric', cvm, cvm < 0.002)

train_eval_probs = gbm.predict(xgb.DMatrix(train_eval[features]))
#train_eval_probs = 0.4*rf.predict_proba(train_eval[features])[:,1] + 0.4*gbm.predict(xgb.DMatrix(train_eval[features])) + 0.2*rf1.predict_proba(train_eval[features])[:,1]

print ('Calculating AUC...')
AUC = evaluation.roc_auc_truncated(train_eval['signal'], train_eval_probs)
print ('AUC XGB', AUC)


print("Make predictions on the test set")
gbmpred = gbm.predict(xgb.DMatrix(test[features]))

print("Make predictions on the test set: xgb")
test_probs = gbmpred
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("xgboost_submission_v1.csv", index=False)