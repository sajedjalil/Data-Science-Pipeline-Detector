# Training parameters copied from https://github.com/yandexdataschool/flavours-of-physics-start/blob/master/flatness_boosting.ipynb

import numpy as np
import pandas as pd
from hep_ml.losses import BinFlatnessLossFunction
from hep_ml.gradientboosting import UGradientBoostingClassifier

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")

print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5])

print("Train a UGradientBoostingClassifier")
loss = BinFlatnessLossFunction(['mass'], n_bins=15, uniform_label=0)
clf = UGradientBoostingClassifier(loss=loss, n_estimators=40, subsample=0.1, 
                                  max_depth=7, min_samples_leaf=10,
                                  learning_rate=0.1, train_features=features, random_state=11)
clf.fit(train[features + ['mass']], train['signal'])

print("Make predictions on the test set")
test_probs = clf.predict_proba(test[features])[:,1]
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("flatness_boosting_submission.csv", index=False)
