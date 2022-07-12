"""

I have modified Ben Hammer's script to accomodate calculation of 
agreement, correlation and weighted roc score before submission.

If you see that its below required threshold then don't submit it.

original author: Ben Hammer.

modifications: Harshaneel Gokhale.

"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import sys
sys.path.append('../input')
import evaluation
import random
print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")
check_agreement = pd.read_csv('../input/check_agreement.csv')
check_correlation = pd.read_csv('../input/check_correlation.csv')
rows = np.random.choice(train.index.values, 4000)
train_eval = train.ix[rows]
train.drop(rows)

print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5])

print("Train a Random Forest model")
rf = ExtraTreesClassifier(n_estimators=400, random_state=1)
rf.fit(train[features], train["signal"])

print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.1,
          "max_depth": 3,
          "min_child_weight": 1,
          "silent": 1,
          "seed": 5}
num_trees=300
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
AUC = roc_auc_score(train_eval['signal'], train_eval_probs)
print ('AUC', AUC)

print("Make predictions on the test set")
test_probs = (rf.predict_proba(test[features])[:,1] +
              gbm.predict(xgb.DMatrix(test[features])))/2
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("rf_xgboost_submission.csv", index=False)