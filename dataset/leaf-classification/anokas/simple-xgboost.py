import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import xgboost as xgb


np.random.seed(42)

train = pd.read_csv('../input/train.csv')
x_train = train.drop(['id', 'species'], axis=1).values
le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])

print(len(x_train))
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=4242)

scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

params = {}
params['booster'] = 'gbtree'
params['objective'] = 'multi:softprob'
params['eval_metric'] = 'mlogloss'
params['eta'] = 0.01
params['silent'] = 1
params['num_class'] = len(le.classes_)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

clf = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=20)

test = pd.read_csv('../input/test.csv')
test_ids = test.pop('id')
x_test = test.values
x_test = scaler.transform(x_test)

y_test = clf.predict(xgb.DMatrix(x_test))

submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
submission.to_csv('submission_log_reg.csv')