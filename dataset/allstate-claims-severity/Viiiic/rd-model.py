# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_csv("../input/train.csv", header=0, sep=',', index_col=0)
X_test = pd.read_csv("../input/test.csv", header=0, sep=',', index_col=0)

X_train = train.drop("loss",axis=1)
Y_train = train['loss']

catFeatureslist = [x for x in train.columns[0:-1] if 'cat' in x]
for cf in catFeatureslist:
    le = LabelEncoder()
    le.fit(X_train[cf].unique())
    X_train[cf] = le.transform(X_train[cf])
    
for cf in catFeatureslist:
    le = LabelEncoder()
    le.fit(X_test[cf].unique())
    X_test[cf] = le.transform(X_test[cf])    


    
RF_model = RandomForestRegressor(criterion='mse', n_jobs = -1)
parameters = {'max_features':[7,15], 
              'n_estimators':[150],
              'min_samples_leaf': [20,50]}

def _score_func(estimator, X, y):
    return mean_absolute_error(y, estimator.predict(X))


clf = GridSearchCV(RF_model, parameters, 
                   cv=KFold(len(Y_train), n_folds = 5, shuffle = True), 
                   scoring=_score_func,
                   verbose=2, refit=True)

clf.fit(X_train, Y_train)


best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])

print(best_parameters)
print(score)

y_predict = clf.predict(X_test)
submit = pd.read_csv("../input/sample_submission.csv")
submit["loss"] = y_predict
submit.to_csv("randomforest_trial.csv", index=False)

# Any results you write to the current directory are saved as output.