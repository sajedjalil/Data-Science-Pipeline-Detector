import numpy as np
## version 2 0.983843

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from keras.utils import np_utils
from hep_ml.losses import BinFlatnessLossFunction
from hep_ml.gradientboosting import UGradientBoostingClassifier

from sklearn.preprocessing import StandardScaler

np.random.seed(671)

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")

print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5])
print("Train a UGradientBoostingClassifier")
loss = BinFlatnessLossFunction(['mass'], n_bins=15, uniform_label=0)
clf = UGradientBoostingClassifier(loss=loss, n_estimators=150, subsample=0.1, # n_estimators = 75
                                  max_depth=7, min_samples_leaf=10,
                                  learning_rate=0.1, train_features=features, random_state=11)
clf.fit(train[features + ['mass']], train['signal'])
fb_preds = clf.predict_proba(test[features])[:,1]
print("Train a Random Forest model")
rf = RandomForestClassifier(n_estimators=250, n_jobs=-1, criterion="entropy", random_state=1)
rf.fit(train[features], train["signal"]) # used to be n_estimators=300, 375 is better, 250 could be fine

print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.2,# used to be 0.2 or 0.1
          "max_depth": 7, # used to be 5 or 6
          "min_child_weight": 1,
          "silent": 1,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=450 #used to be 300, 375 is better
gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)

print("Make predictions on the test set")
# test_probs = (0.35*rf.predict_proba(test[features])[:,1]) + (0.35*gbm.predict(xgb.DMatrix(test[features])))+(0.15*predskeras) + (0.15*fb_preds) 
test_probs = (0.24*rf.predict_proba(test[features])[:,1]) + (0.3*gbm.predict(xgb.DMatrix(test[features])))+ (0.20*fb_preds) #is better
# test_probs = (0.25*rf.predict_proba(test[features])[:,1]) + (0.25*gbm.predict(xgb.DMatrix(test[features])))+(0.25*predskeras) + (0.25*fb_preds)
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("rf_xgboost_keras_flatness_v5.csv", index=False)