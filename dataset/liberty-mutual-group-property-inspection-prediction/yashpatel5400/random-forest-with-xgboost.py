"""
Author: Yash Patel
Name: LibertyMutual.py 
Description: Predicts the hazard score based on provided
anonymized columns using Random Forest with XGBoost
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb

train = pd.read_csv('../input/train.csv', index_col='Id')
test = pd.read_csv('../input/test.csv', index_col='Id')

columns = train.drop(['Hazard'], axis=1).columns
for column in columns:
    uniqueVals = train[column].unique()
    if not isinstance(uniqueVals[0], int):
        mapper = dict(zip(uniqueVals, range(len(uniqueVals))))
        train[column] = train[column].map(mapper).astype(int)
        test[column] = test[column].map(mapper).astype(int)

# print("Train a Gradient Boosting model")
# clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.005, subsample=0.7,
#                                      min_samples_leaf=10, max_depth=7, random_state=11)
print("Train a Random Forest model")
clf = RandomForestClassifier(n_estimators=25)

clf.fit(train[columns], train['Hazard'])

print("Train a XGBoost model")
params = {"objective": "reg:linear",
          "eta": 0.3,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=250
gbm = xgb.train(params, xgb.DMatrix(
    train[columns], train['Hazard']), num_trees)

print("Make predictions on the test set")
test_probs = (clf.predict_proba(test[columns])[:,1] +
              gbm.predict(xgb.DMatrix(test[columns])))/2

result = pd.DataFrame({'id': test.index})
result['Hazard'] = clf.predict_proba(test[columns])[:, 1]
result.to_csv('result.csv', index=False, sep=',')