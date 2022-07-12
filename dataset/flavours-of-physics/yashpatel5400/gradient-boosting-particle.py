"""
Author: Yash Patel
Name: PhysicsPredict.py 
Description: Predicts, using random forest machine learning,
the likelihood of a signal based on the lifetime of a particle
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb

train = pd.read_csv('../input/training.csv', index_col='id')

print("Training gradient boosted model...")
variables = train.columns[0:-5]

# Attempt to use only the most significant variables determined
#variables = ['IPSig', 'ISO_SumBDT', 'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof', \
#    'LifeTime', 'VertexChi2', 'iso', 'p0_IPSig', 'dira', 'IP', 'p1_eta']
#clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.005, subsample=0.7,
#                                      min_samples_leaf=10, max_depth=7, random_state=11)
clf = RandomForestClassifier(n_estimators=25)

clf.fit(train[variables], train['signal'])

print("XGBoost model...")
params = {"objective": "binary:logistic",
          "eta": 0.3,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
          
num_trees=250
gbm = xgb.train(params, xgb.DMatrix(
    train[variables], train['signal']), num_trees)

print("Make predictions on the test set")

test = pd.read_csv('../input/test.csv', index_col='id')
test['id'] = test.index
test_probs = (clf.predict_proba(test[variables])[:,1] +
              gbm.predict(xgb.DMatrix(test[variables])))/2
result = pd.DataFrame({'id': test['id'], "prediction": test_probs})
result.to_csv('result.csv', index=False, sep=',')