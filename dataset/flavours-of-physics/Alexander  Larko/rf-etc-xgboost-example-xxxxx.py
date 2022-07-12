import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier

import xgboost as xgb

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")

print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5])

print("Train a rf model")

#etc = ExtraTreesClassifier(n_estimators=500, n_jobs=-1,criterion="entropy", random_state=11)
#etc.fit(train[features], train["signal"])
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1,criterion="entropy", random_state=11)
rf.fit(train[features], train["signal"])
print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.1,
          "max_depth": 7,
          "min_child_weight": 10,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=500
gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)

print("Make predictions on the test set")
test_probs1 = rf.predict_proba(test[features])[:,1]
test_probs2 = gbm.predict(xgb.DMatrix(test[features]))
test_probs=test_probs1*0.4+test_probs2*0.6

submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("rf_xgboost_submission.csv", index=False)