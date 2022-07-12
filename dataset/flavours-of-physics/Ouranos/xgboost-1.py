import numpy as np
np.random.seed(123)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")

print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5])
labels = train.signal
ids = test.id
train = train[features]
test = test[features]

train, scaler = preprocess_data(train)
test, scaler = preprocess_data(test, scaler)

print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.05,
          "max_depth": 5,
          "min_child_weight": 4,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=500
gbm = xgb.train(params, xgb.DMatrix(train, labels), num_trees)

print("Make predictions on the test set")
test_probs = gbm.predict(xgb.DMatrix(test))
submission = pd.DataFrame({"id": ids, "prediction": test_probs})
submission.to_csv("xgboost.csv", index=False)