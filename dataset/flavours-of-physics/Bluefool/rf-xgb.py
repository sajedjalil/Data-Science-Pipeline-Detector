import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")
train.drop(["IPSig"], axis=1)
test.drop(["IPSig"], axis=1)
print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5])

print("Train a Random Forest model")
clf =KNeighborsClassifier(n_neighbors=10)
clf.fit(train[features], train["signal"])

print("Make predictions on the test set")
test_probs = clf.predict_proba(test[features])[:,1] 
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("rf_xgboost_submission.csv", index=False)