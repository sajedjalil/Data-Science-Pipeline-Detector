import numpy as np
import pandas as pd
from datetime import datetime

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

dropfeats = ['count', 'casual', 'registered']

train['hour'] = train['datetime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
train['day'] = train['datetime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').day)
train['weekday'] = train['datetime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday())
train['month'] = train['datetime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month)
train['year'] = train['datetime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').year)
train['count'] = np.log1p(train['count'])

train = train.drop('datetime', axis=1)

test['hour'] = test['datetime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
test['day'] = test['datetime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').day)
test['weekday'] = test['datetime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday())
test['month'] = test['datetime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month)
test['year'] = test['datetime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').year)

timecolumn = test['datetime']

test = test.drop('datetime', axis=1)

X = train.drop(dropfeats, axis=1).values
Y = train['count'].values
testX = test.values

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer

#Optimization part

#def loss_func(truth, prediction):
#    y = np.expm1(truth)
#    y_ = np.expm1(prediction)
#    log1 = np.array([np.log(x + 1) for x in truth])
#    log2 = np.array([np.log(x + 1) for x in prediction])
#    return np.sqrt(np.mean((log1 - log2)**2))
    
#param_grid = {
#    'learning_rate': [0.1, 0.01, 0.001, 0.0001],
#    'n_estimators': [100, 1000, 1500, 2000, 4000],
#    'max_depth': [1, 2, 3, 4, 5, 8, 10]
#}

#scorer = make_scorer(loss_func, greater_is_better=False)
#model = GradientBoostingRegressor(random_state=42)
#result = GridSearchCV(model, param_grid, cv=4, scoring=scorer, n_jobs=3).fit(X, Y)
#print('\tParams:', result.best_params_)
#print('\tScore:', result.best_score_)

#That will print 
#Params:', {'n_estimators': 2000, 'learning_rate': 0.01, 'max_depth': 4}
#Score:', -0.09649149584358846

gbr = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.01, max_depth=4).fit(X, Y)

pred = gbr.predict(testX)
pred = np.expm1(pred)
submission = pd.DataFrame({
        "datetime": timecolumn,
        "count": pred
    })
submission.to_csv('GBR.csv', index=False)