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
import xgboost as xgb
import sys
sys.path.append('../input')
import evaluation

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")
check_agreement = pd.read_csv('../input/check_agreement.csv')
check_correlation = pd.read_csv('../input/check_correlation.csv')
train_eval = train[train['min_ANNmuon'] > 0.4]

print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5])

print("Train a Random Forest model")
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, criterion="entropy", random_state=1)
rf.fit(train[features], train["signal"])

print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.1,
          "max_depth": 4,
          "min_child_weight": 9,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=500
gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)

agreement_probs= (rf.predict_proba(check_agreement[features])[:,1] +
	gbm.predict(xgb.DMatrix(check_agreement[features])))/2
print('Checking agreement...')
ks = evaluation.compute_ks(
	agreement_probs[check_agreement['signal'].values == 0],
	agreement_probs[check_agreement['signal'].values == 1],
	check_agreement[check_agreement['signal'] == 0]['weight'].values,
	check_agreement[check_agreement['signal'] == 1]['weight'].values)
print ('KS metric', ks, ks < 0.09)

correlation_probs = (rf.predict_proba(check_correlation[features])[:,1] + 
	gbm.predict(xgb.DMatrix(check_correlation[features])))/2
print ('Checking correlation...')
cvm = evaluation.compute_cvm(correlation_probs, check_correlation['mass'])
print ('CvM metric', cvm, cvm < 0.002)

train_eval_probs = (rf.predict_proba(train_eval[features])[:,1] +
	gbm.predict(xgb.DMatrix(train_eval[features])))/2
print ('Calculating AUC...')
AUC = evaluation.roc_auc_truncated(train_eval['signal'], train_eval_probs)
print ('AUC', AUC)

print("Make predictions on the test set")
test_probs = (rf.predict_proba(test[features])[:,1] +
              gbm.predict(xgb.DMatrix(test[features])))/2
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("rf_xgboost_submission.csv", index=False)