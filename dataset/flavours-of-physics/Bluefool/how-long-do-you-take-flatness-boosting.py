# Training parameters copied from https://github.com/yandexdataschool/flavours-of-physics-start/blob/master/flatness_boosting.ipynb

import numpy as np
import pandas as pd
from hep_ml.losses import BinFlatnessLossFunction, KnnFlatnessLossFunction
from hep_ml.gradientboosting import UGradientBoostingClassifier

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")
train2 = train
train2 = train2.drop(["IPSig"], axis=1)
print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5])
features2 = list(train2.columns[1:-5])
uniform_features  = ["mass"]

print("Train a UGradientBoostingClassifier")
loss = KnnFlatnessLossFunction(uniform_features, uniform_label=0)
clf = UGradientBoostingClassifier(loss=loss, n_estimators=150,  
                                  max_depth=6,
                                  learning_rate=0.15, train_features=features, random_state=369)
clf.fit(train[features + ['mass']], train['signal'])


loss = BinFlatnessLossFunction(uniform_features, uniform_label=0)
binugb = UGradientBoostingClassifier(loss=loss, n_estimators=150,
                                  max_depth=6, learning_rate=0.15, train_features=features2, random_state=369)
binugb.fit(train2[features2 + ['mass']], train2['signal'])

print("Make predictions on the test set")
test_probs = (clf.predict_proba(test[features])[:,1]+binugb.predict_proba(test[features2])[:,1])/2
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("flatness_boosting_v20.csv", index=False)
