# import library and data
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_squared_log_error, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

train = pd.read_csv('../input/bike-sharing-demand/train.csv', engine='python')
test = pd.read_csv('../input/bike-sharing-demand/test.csv', engine='python')
submission = pd.read_csv('../input/bike-sharing-demand/sampleSubmission.csv', engine='python')

copied = train.copy()
X_train_copied = train.copy().drop(['count'], axis=1)
y_train_copied = train.copy()['count']

# print(train.head())
# print(test.head())

# Drop casual and registered - just a decomposition of count
train.drop(['casual', 'registered'], axis=1, inplace=True)

# Check Correlation and Distribution
# print(train.corr())
# print(train.describe())

# For datetime column, split it into year-month-day-time; seems that the month and time is much more important than the others
train['datetime'] = pd.to_datetime(train['datetime'])
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour

test['datetime'] = pd.to_datetime(test['datetime'])
test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day
test['hour'] = test['datetime'].dt.hour

#### 전처리 ####

# check monthly difference
# print(train[['month', 'count']].groupby(by='month').sum()) # month seems to be an important feature

# Split into numberical_features and categorical_features
# print(train.columns)
num_features = ['temp', 'atemp', 'humidity', 'windspeed', 'count']
cat_features = ['season', 'holiday', 'workingday', 'weather']

# Check skewness
# for num in num_features:
#     print(num, ':', train[num].skew())
    
# Take a log-transformation on count, then check skewness again
train['count'] = np.log1p(train['count'])
# print(num, ':', train['count'].skew())

# Since year can be regarded as a categorical variable, subtract 2011 from year
# and drop datetime, day, and temp columns - since atemp is more important than the temp
train['year'] = train['year'] - train['year'].min()
test['year'] = test['year'] - test['year'].min()

train.drop(['datetime', 'day', 'temp'], axis=1, inplace=True)
test.drop(['datetime', 'day', 'temp'], axis=1, inplace=True)

# X, y split
X_train = train.drop(['count'], axis=1)
y_train = train['count'] # transformed

#### 모델 선정 및 학습 ####

# model selection
# rf = RandomForestRegressor(random_state=42)
# gbr = GradientBoostingRegressor(random_state=42)
xgb = XGBRegressor(random_state=42)

# models = [rf, gbr, xgb]
# for model in models:
#     name = model.__class__.__name__
#     neg_mse = cross_val_score(model, X=X_train, y=y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
#     rmse = np.sqrt((-1) * np.mean(neg_mse))
#     print('Model: %s, RMSE: %.4f' % (name, rmse)) # best model - XGB
    
# # GridSearch
# params = {
#     'n_estimators': [100, 300, 500],
#     'max_depth': [3, 5, 7],
#     'n_jobs': [-1],
#     'random_state': [42]
# }
# search = GridSearchCV(xgb, param_grid=params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
# search.fit(X_train, y_train)
# best_params = search.best_params_
# best_score = np.sqrt((-1) * np.mean(search.best_score_))
# print('Best params: {}\nBest score: {}'.format(best_params, round(best_score, 4)))

# fitting and prediction
best_params = {'max_depth': 3, 'n_estimators': 300, 'n_jobs': -1, 'random_state': 42}
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=42)
xgb_final = XGBRegressor(**best_params)
xgb_final.fit(X_tr, y_tr)

val_pred = xgb_final.predict(X_val)
val_mse = mean_squared_error(y_val, val_pred)
print('RMSE on the log-transformed target: %.4f' % np.sqrt(val_mse)) # 0.3219

y_pred = xgb_final.predict(test)
real_pred = np.expm1(y_pred)
print(real_pred)

submission['count'] = real_pred
print(submission.head())


    
    
    














