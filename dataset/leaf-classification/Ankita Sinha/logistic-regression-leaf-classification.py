# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import xgboost as xgb

np.random.seed(42)
#data is highle colinear  found out while running lda :p
train = pd.read_csv('../input/train.csv')
x_train = train.drop(['id', 'species'], axis=1).values
#to encode label
le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
#plt.scatter(x_train['margin1'],y_train)
plt.show()
test = pd.read_csv('../input/test.csv')
test_ids = test.pop('id')
x_test = test.values
#(x-mean)/std-deviation
scaler = StandardScaler().fit(x_test)
x_test = scaler.transform(x_test)

params = {'C':[1, 10, 50, 100, 950,1500], 'tol': [0.001, 0.008, 0.007]}
log_reg = LogisticRegression(solver='lbfgs', multi_class='multinomial')
clf = GridSearchCV(log_reg,params, scoring='log_loss', refit='False', n_jobs=1, cv=5)
#clf = xgb.XGBClassifier(silent=False,  learning_rate=0.03)
clf.fit(x_train, y_train)
y_test = clf.predict_proba(x_test)
#print((clf.score(x_train,y_train)))
print('Best parameters: {}'.format(clf.best_params_))
submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
submission.to_csv('submission.csv')