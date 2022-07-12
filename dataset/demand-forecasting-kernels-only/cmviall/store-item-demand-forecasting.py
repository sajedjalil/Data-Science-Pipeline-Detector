# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_path = '/kaggle/input/demand-forecasting-kernels-only/train.csv'
test_path = '/kaggle/input/demand-forecasting-kernels-only/test.csv'

output_variable = 'sales'

X = pd.read_csv(train_path)
X_test = pd.read_csv(test_path, index_col='id', sep=',')
X.dropna(axis=0, subset=[output_variable], inplace=True)
y = X.sales
X.drop([output_variable], axis=1, inplace=True)

cal = calendar()
holidays = cal.holidays(start=X['date'].min(), end=X['date'].max())
X['holiday'] = X['date'].isin(holidays)
X['holiday'] = X['holiday'].astype(int)

holidays = cal.holidays(start=X_test['date'].min(), end=X_test['date'].max())
X_test['holiday'] = X_test['date'].isin(holidays)
X_test['holiday'] = X_test['holiday'].astype(int)
X['date'] = pd.to_datetime(X['date'], infer_datetime_format=True)
dates = X['date']

X = X.assign(year=dates.dt.year,
                           month=dates.dt.month.astype('uint8'),
                           day=dates.dt.day.astype('uint8'),
                           day_of_week=dates.dt.dayofweek,
                           )

X.drop(['date'], axis=1, inplace=True)
print('1')

X_test['date'] = pd.to_datetime(X_test['date'], infer_datetime_format=True)
date = X_test['date']

X_test = X_test.assign(year=date.dt.year,
                           month=date.dt.month.astype('uint8'),
                           day=date.dt.day.astype('uint8'),
                           day_of_week=date.dt.dayofweek,
                           )

X_test.drop(['date'], axis=1, inplace=True)

X.loc[X['day'] > 24, 'end_of_month'] = 1
X.loc[X['day'] <= 24, 'end_of_month'] = 0
X.loc[X['day'] < 7, 'beginning_of_month'] = 1
X.loc[X['day'] >= 7, 'beginning_of_month'] = 0
X_test.loc[X_test['day'] > 24, 'end_of_month'] = 1
X_test.loc[X_test['day'] <= 24, 'end_of_month'] = 0
X_test.loc[X_test['day'] < 7, 'beginning_of_month'] = 1
X_test.loc[X_test['day'] >= 7, 'beginning_of_month'] = 0

train_day = pd.get_dummies(X['day'], prefix_sep='_', drop_first=True)
train_day_week = pd.get_dummies(X['day_of_week'], prefix_sep='_', drop_first=True, columns=['sun','mon','tues','wed','thurs','fri','sat'])
X.drop(['day', 'day_of_week'], axis=1, inplace=True)
X = pd.concat([X,train_day, train_day_week], axis=1)
test_day = pd.get_dummies(X_test['day'], prefix_sep='_', drop_first=True)
test_day_week = pd.get_dummies(X_test['day_of_week'], prefix_sep='_', drop_first=True, columns=['sun','mon','tues','wed','thurs','fri','sat'])
X_test.drop(['day', 'day_of_week'], axis=1, inplace=True)
X_test = pd.concat([X_test,test_day, test_day_week], axis=1)

print(X)
print(X_test)

# Split validation data from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=0)

# Declare randomforestregressor model
model = RandomForestRegressor(n_estimators=100, random_state=0)
clf = model
        
# Fitting model to train data via pipeline
clf.fit(X_train, y_train)

# Predict outcomes from validation data
preds = clf.predict(X_valid)

# Predict outcomes from test data
preds_test = clf.predict(X_test)

# Print mean absolute error for validation data
print('Validation MAE:', mean_absolute_error(y_valid, preds))

print(X_test.index)
output = pd.DataFrame({'id': X_test.index, 'sales': preds_test})
output.to_csv('/kaggle/working/submission.csv', index=False)
print(output)
