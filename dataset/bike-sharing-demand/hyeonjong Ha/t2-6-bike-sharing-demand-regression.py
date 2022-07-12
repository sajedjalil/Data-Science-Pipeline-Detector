import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

train = pd.read_csv('../input/bike-sharing-demand/train.csv')
test = pd.read_csv('../input/bike-sharing-demand/test.csv')

train['hour'] = train['datetime'].str[11:13]
test['hour'] = test['datetime'].str[11:13]

train_x_raw = train.drop(['datetime', 'casual', 'registered', 'count'], axis=1)
train_y = train['count']

test_x_raw = test.drop(['datetime'], axis=1)
test_y = test['datetime']

total_x = pd.concat([train_x_raw, test_x_raw])
total_x[['season', 'weather']] = total_x[['season', 'weather']].astype('str')
total_x = pd.get_dummies(total_x, ['hour', 'season', 'weather'])

train_x = total_x[:10886]
test_x = total_x[10886:]

tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y, test_size=0.7, random_state=777)

model = XGBRegressor(n_estimators=50, max_depth=50)
model.fit(tr_x, tr_y)
print(model.score(tr_x, tr_y))
print(model.score(val_x, val_y))
print(np.sqrt(mean_squared_log_error(np.abs(model.predict(val_x)), val_y)))

result = test_y.copy()
result['count'] = model.predict(test_x)
result.to_csv('0000.csv', index=False)

# 결과값(RMSLE) : 0.59

# n_estimators = [10, 50, 100]
# max_depths = [10, 50, 100]

# for n_estimator in n_estimators: # 100, 50, 0.78, 0.60
#     for max_depth in max_depths:
#         print(n_estimator, max_depth)
#         model = RandomForestRegressor(n_estimators=n_estimator, max_depth=max_depth)
#         model.fit(tr_x, tr_y)
#         print(model.score(tr_x, tr_y))
#         print(model.score(val_x, val_y))
#         print(np.sqrt(mean_squared_log_error(model.predict(val_x), val_y)))
        
# for n_estimator in n_estimators:    #50, 50, 0.77, 0.59
#     for max_depth in max_depths:
#         print(n_estimator, max_depth)
#         model = XGBRegressor(n_estimators=n_estimator, max_depth=max_depth)
#         model.fit(tr_x, tr_y)
#         print(model.score(tr_x, tr_y))
#         print(model.score(val_x, val_y))
#         print(np.sqrt(mean_squared_log_error(np.abs(model.predict(val_x)), val_y)))
        
        
        