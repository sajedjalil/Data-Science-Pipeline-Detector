# import library and data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

def exam_data_load(df, target, id_name="", null_name=""):
    if id_name == "":
        df = df.reset_index().rename(columns={"index": "id"})
        id_name = 'id'
    else:
        id_name = id_name
    
    if null_name != "":
        df[df == null_name] = np.nan
    
    X_train, X_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=2021)
    y_train = X_train[[id_name, target]]
    X_train = X_train.drop(columns=[id_name, target])
    y_test = X_test[[id_name, target]]
    X_test = X_test.drop(columns=[id_name, target])
    return X_train, X_test, y_train, y_test 

df = pd.read_csv("../input/bike-sharing-demand/train.csv")
X_train, X_test, y_train, y_test = exam_data_load(df, target='count')

#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#print(X_train.head())
#print(X_test.head())
#print(X_train.info)
# 결측치 없음
#print(X_train.isnull().sum())
#print(X_test.isnull().sum())

X_train.drop('datetime', axis = 1, inplace = True)
X_test.drop('datetime', axis = 1, inplace = True)
#print(X_train.head(), X_test.head())

scaler = StandardScaler()
num_features  =['temp','atemp','humidity','windspeed']
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])
#print(X_train.head(), X_test.head())

    
X_tr, X_ts, y_tr, y_ts = train_test_split(X_train, y_train['count'], test_size = 0.2, random_state = 2002)

rf = RandomForestRegressor()
rf.fit(X_tr, y_tr)
rf_pred = rf.predict(X_ts)
rf_mse = mean_squared_error(y_ts, rf_pred)
round(np.sqrt(rf_mse), 3)


xg = XGBRegressor(max_depth = 6, n_estimators = 100)
xg.fit(X_tr, y_tr)
xg_pred = xg.predict(X_ts)
xg_mse = mean_squared_error(y_ts, xg_pred)
round(np.sqrt(xg_mse), 3)

# GridSearch
#params = {
#    'n_estimators': [50, 75, 100],
#    'max_depth': [3, 5, 7],
#    'n_jobs': [-1],
#    'random_state': [2002]
#}
#search = GridSearchCV(xg, param_grid=params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
#search.fit(X_tr, y_tr)    
#best_params = search.best_params_
#best_score = np.sqrt((-1) * np.mean(search.best_score_))
#print('Best params: {}\nBest score: {}'.format(best_params, round(best_score, 4)))

# fitting and prediction
best_params = {'max_depth': 7, 'n_estimators': 500, 'n_jobs': -1, 'random_state': 2002}
#X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=202)
rf_final = XGBRegressor(**best_params)
rf_final.fit(X_train, y_train['count'])

pred = xg.predict(X_test)
xg_mse = mean_squared_error(y_test['count'], pred)
np.sqrt(xg_mse)

sub = pd.DataFrame({'id': y_test['id'], 'count' : pred})
sub.to_csv('2020.csv', index = False)

pd.read_csv('./2020.csv')


