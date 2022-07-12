import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")

print("Dropping James King's fields")
train.drop(['p0_p','p1_p','p2_p','p0_eta','p1_eta','p2_eta','SPDhits'], axis=1, inplace=True)
test.drop(['p0_p','p1_p','p2_p','p0_eta','p1_eta','p2_eta','SPDhits'], axis=1, inplace=True)

features = list(train.columns[1:-4])
print(features)
print(test[features])
print("Train a Random Forest model")
rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, criterion="entropy", random_state=11)
rf.fit(train[features], train["signal"])

print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.15,
          "max_depth": 7,
          "min_child_weight": 10,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=300
gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)

print("Make predictions on the test set")
test_probs = (rf.predict_proba(test[features])[:,1] +
              gbm.predict(xgb.DMatrix(test[features])))/2
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("rf_xgboost_james_fields_submission.csv", index=False)