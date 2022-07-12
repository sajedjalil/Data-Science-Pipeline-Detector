import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import sys

#import evaluation
exec(open("../input/evaluation.py").read())
from sklearn.cross_validation import *

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")

#have some feature engineering work for better rank
#def add_features(df):
#    #significance of flight distance
#    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
#    return df
    
#print("Adding features to both training and testing")
#train = add_features(train)
#test = add_features(test)
    
print("Loading check agreement for KS test evaluation")
check_agreement = pd.read_csv('../input/check_agreement.csv')
check_correlation = pd.read_csv('../input/check_correlation.csv')
#check_agreement = add_features(check_agreement)
#check_correlation = add_features(check_correlation)
train_eval = train[train['min_ANNmuon'] > 0.4]
    
print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5])
print("features:",features)

print("Train a Random Forest model")
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(train[features], train["signal"])

print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.3,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=250
gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)

test_agreement_ = True
if test_agreement_:
    agreement_probs= (rf.predict_proba(check_agreement[features])[:,1] + gbm.predict(xgb.DMatrix(check_agreement[features])))/2

    ks = compute_ks(
            agreement_probs[check_agreement['signal'].values == 0],
            agreement_probs[check_agreement['signal'].values == 1],
            check_agreement[check_agreement['signal'] == 0]['weight'].values,
            check_agreement[check_agreement['signal'] == 1]['weight'].values)
    print ('KS metric', ks, ks < 0.09)

    
    
print("Make predictions on the test set")
test_probs = (rf.predict_proba(test[features])[:,1] +
              gbm.predict(xgb.DMatrix(test[features])))/2
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("rf_xgboost_submission.csv", index=False)