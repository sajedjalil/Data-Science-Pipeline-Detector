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

print("Turning on random forests...")
from sklearn.ensemble import RandomForestRegressor

import resource

rf = RandomForestRegressor(n_jobs=-1, n_estimators=2000)
rf.fit(X_train, y_train)

print("Predicting...")

preds = rf.predict(X_test)
preds = pd.DataFrame({'Hazard': preds, 'id': test.index})
preds.to_csv('benchmark.csv', index=False)

print("Memory usage: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000, "MB")