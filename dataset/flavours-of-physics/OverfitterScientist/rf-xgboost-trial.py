import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")

print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5])

print("Train a Random Forest model")
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, criterion="entropy", random_state=1)
rf.fit(train[features], train["signal"])

# Create and fit an AdaBoosted decision Tree
print("Train a Adaboost model")
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

bdt.fit(train[features], train["signal"])

print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.3,
          "max_depth": 5,
          "min_child_weight": 10,
          "silent": 1,
          "subsample": 0.5,
          "colsample_bytree": 0.7,
          "seed": 1}
          
num_trees=1000
gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)

print("Make predictions on the test set")
test_probs = ((rf.predict_proba(test[features])[:,1] + bdt.predict_proba(test[features])[:,1] +
              gbm.predict(xgb.DMatrix(test[features])))/3)
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("rf_xgboost_submission.csv", index=False)