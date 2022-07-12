import pandas as pd
import numpy as np

print("Reading data...")
# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)
train = pd.concat([train, test])

# preprocessing columns
# factorizing column values
train['T1_V4'] = pd.factorize(train['T1_V4'])[0]
train['T1_V5'] = pd.factorize(train['T1_V5'])[0]
# simple yes/no
train['T1_V6'] = pd.factorize(train['T1_V6'])[0]
train['T1_V7'] = pd.factorize(train['T1_V7'])[0]
train['T1_V8'] = pd.factorize(train['T1_V8'])[0]
train['T1_V9'] = pd.factorize(train['T1_V9'])[0]
train['T1_V11'] = pd.factorize(train['T1_V11'])[0]
train['T1_V12'] = pd.factorize(train['T1_V12'])[0]
train['T1_V15'] = pd.factorize(train['T1_V15'])[0]
train['T1_V16'] = pd.factorize(train['T1_V16'])[0]
train['T1_V17'] = pd.factorize(train['T1_V17'])[0]

train['T2_V3'] = pd.factorize(train['T2_V3'])[0]
train['T2_V5'] = pd.factorize(train['T2_V5'])[0]
train['T2_V11'] = pd.factorize(train['T2_V11'])[0]
train['T2_V12'] = pd.factorize(train['T2_V12'])[0]
train['T2_V13'] = pd.factorize(train['T2_V13'])[0]
train, test = train[train.Hazard.notnull()], train[train.Hazard.isnull()]
train_target = train.Hazard

train.drop('Hazard', axis=1, inplace=True)
test.drop('Hazard', axis=1, inplace=True)

X_train = np.array(train)
y_train = np.array(train_target)
X_test = np.array(test)

print("Turning on xgboost...")
import xgboost as xgb

import resource
#{'max_delta_step': 5, 'learning_rate': 0.05, 'n_estimators': 300, 'max_depth': 4,
#'min_child_weight': 10, 'gamma': 5}
gbm = xgb.XGBRegressor(max_depth=4, objective="reg:linear",
                       n_estimators=300, learning_rate=0.05,
                       min_child_weight=10, gamma=5,
                       max_delta_step=5).fit(X_train, y_train)


print("Predicting...")

preds = gbm.predict(X_test)
preds = pd.DataFrame({'Hazard': preds, 'id': test.index})
preds.to_csv('benchmark.csv', index=False)

print("Memory usage:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000, "MB")