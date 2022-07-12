import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")

print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5])

print("Train a Random Forest model")
rf = RandomForestClassifier(n_estimators=90, random_state=0)
rf.fit(train[features], train["signal"])

print("Train a XGBoost model")
params = {"objective": "reg:logistic",
          "eta": 0.4,
          "gamma": 5,
          "max_depth": 3,
          "min_child_weight": 5,
          "silent": 1.5,
          "alpha": 0.2,
          "seed": 586}
num_trees=50
gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)

print("Make predictions on the test set")
test_probs = (rf.predict_proba(test[features])[:,1] +
              gbm.predict(xgb.DMatrix(test[features]))*2)/3
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("rf_xgboost_submission.csv", index=False)