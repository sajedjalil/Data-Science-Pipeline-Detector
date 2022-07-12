import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")

features = list(train.columns[1:-5])
print("Train an Adaboost model")
et = AdaBoostRegressor(n_estimators=100, random_state=369)
et.fit(train[features], train["signal"])

print("Make predictions on the test set")
test_probs = et.predict(test[features])
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("ad_v1.csv", index=False)