import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")

features = list(train.columns[1:-5])
print("Train a Random Forest model")
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=369)
rf.fit(train[features], train["signal"])

print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.15,
          "max_depth": 3,
          "min_child_weight": 1,
          "silent": 1,
          "seed": 369}
num_trees=1000
gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)

print("Make predictions on the test set")
test_probs = (rf.predict_proba(test[features])[:,1] +
              gbm.predict(xgb.DMatrix(test[features])))/2
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("rf_xgboost_tune1.csv", index=False)