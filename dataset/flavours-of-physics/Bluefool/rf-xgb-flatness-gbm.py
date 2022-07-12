# version 4 0.982448
import numpy as np
np.random.seed(369)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from hep_ml.losses import BinFlatnessLossFunction,KnnFlatnessLossFunction
from hep_ml.gradientboosting import UGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")
np.random.seed(369)
uniform_features=['mass']
#print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5])
print("Train a UGradientBoostingClassifier")
loss = BinFlatnessLossFunction(['mass'], n_bins=15, uniform_label=0)
clf = UGradientBoostingClassifier(loss=loss, n_estimators=100, 
                                  max_depth=5,
                                  learning_rate=0.1, train_features=features, random_state=369)
clf.fit(train[features + ['mass']], train['signal'])
fb_preds = clf.predict_proba(test[features])[:,1]

loss = KnnFlatnessLossFunction(uniform_features, uniform_label=0)
clf = UGradientBoostingClassifier(loss=loss, n_estimators=100,  
                                  max_depth=5,
                                  learning_rate=0.1, train_features=features, random_state=369)
clf.fit(train[features + ['mass']], train['signal'])
kn_preds = clf.predict_proba(test[features])[:,1]

print("Train a Random Forest model")
np.random.seed(369)

rf = RandomForestClassifier(n_estimators=200, n_jobs=-1,random_state=369)
rf.fit(train[features], train["signal"])

print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.2,
          "max_depth": 5,
          "min_child_weight": 1,
          "silent": 1,
          "seed": 369}
num_trees=200
np.random.seed(369)

gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)

print("Make predictions on the test set")
test_probs = (0.25*kn_preds)+(0.3*rf.predict_proba(test[features])[:,1]) + (0.3*gbm.predict(xgb.DMatrix(test[features]))) + (0.25*fb_preds)
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("rf_xgboost_fn_knn.csv", index=False)