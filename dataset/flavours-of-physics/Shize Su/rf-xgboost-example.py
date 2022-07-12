import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")

print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5])

print("Train a Random Forest model")
rf = RandomForestClassifier(n_estimators=500, random_state=8)
rf.fit(train[features], train["signal"])


print("Train a Random Forest model")
et =  ensemble.ExtraTreesClassifier(n_estimators=400, random_state=8)
#et.fit(train[features], train["signal"])

print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.3,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=500
#gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)

print("Make predictions on the test set")
#test_probs = (rf.predict_proba(test[features])[:,1] +
              #gbm.predict(xgb.DMatrix(test[features])))/2
              
#test_probs = (0.5*rf.predict_proba(test[features])[:,1]+0.5*et.predict_proba(test[features])[:,1])
              
test_probs = (rf.predict_proba(test[features])[:,1])
              
                       
              
              
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("rf_500_seed8.csv", index=False)