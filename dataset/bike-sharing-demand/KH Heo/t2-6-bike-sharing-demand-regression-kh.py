# 시험환경 세팅 (코드 변경 X)
import pandas as pd
import numpy as np
  
train = pd.read_csv("../input/bike-sharing-demand/train.csv", parse_dates=['datetime'])
test = pd.read_csv("../input/bike-sharing-demand/test.csv", parse_dates=['datetime'])
print(train.columns)
print(test.columns)
print('-' * 80)

## Preprocessing
train['datetime-year'] = train['datetime'].dt.year
train['datetime-month'] = train['datetime'].dt.month
train['datetime-day'] = train['datetime'].dt.day
train['datetime-hour'] = train['datetime'].dt.hour
train['datetime-dayofweek'] = train['datetime'].dt.dayofweek
test['datetime-year'] = test['datetime'].dt.year
test['datetime-month'] = test['datetime'].dt.month
test['datetime-day'] = test['datetime'].dt.day
test['datetime-hour'] = test['datetime'].dt.hour
test['datetime-dayofweek'] = test['datetime'].dt.dayofweek

print(train.shape)
print('-' * 80)

## missing value
print(train.isnull().sum(), test.isnull().sum())

## feature seletion
print(train.columns)
features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed',
            'datetime-year', 'datetime-hour', 'datetime-dayofweek']
X_train = train[features]
y_train = train['count']
X_test = test[features]

## modeling
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

rf = RandomForestRegressor(random_state=777, n_jobs=-1)
score = cross_val_score(rf,
                        X_train,
                        y_train,
                        cv=5,
                        scoring='neg_mean_squared_error')
print(score.mean() * -1)

## Hyper parameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators' : [50, 100 ,150],
              'max_depth' : ['None', 5, 7]}

rf_grid = GridSearchCV(rf,
                       param_grid = param_grid,
                       cv=3,
                       n_jobs=-1,
                       scoring='neg_mean_squared_error')
rf_grid.fit(X_train, y_train)

print('best_estimators: ', rf_grid.best_estimator_)
print('best_parameters: ', rf_grid.best_params_)
print('best_score: ', rf_grid.best_score_ * -1)





