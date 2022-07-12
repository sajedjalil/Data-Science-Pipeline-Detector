import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")
wtf = np.sum(train["signal"]) * 1.0
#pos = (train["signal"].shape[0] - wtf) / wtf
pos = 0.6
print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5])

print("Train a Random Forest model")
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, criterion="entropy", random_state=456)
rf.fit(train[features], train["signal"])

print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.05,
          "max_depth": 8,
          'scale_pos_weight': pos,
          "min_child_weight": 15,
          "silent": 1,
          "subsample": 0.5,
          "colsample_bytree": 0.5,
          "seed": 789}
num_trees=500
gbm1 = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)
num_trees=750
params = {"objective": "binary:logistic",
          "eta": 0.05,
          "max_depth": 7,
          'scale_pos_weight': pos,
          "min_child_weight": 10,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1024}
gbm2 = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)
print("Make predictions on the test set")
test_probs = (rf.predict_proba(test[features])[:,1] +
              gbm1.predict(xgb.DMatrix(test[features])) + 
              gbm2.predict(xgb.DMatrix(test[features]))) / 3.0
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("rf_xgboost_submission.csv", index=False)
