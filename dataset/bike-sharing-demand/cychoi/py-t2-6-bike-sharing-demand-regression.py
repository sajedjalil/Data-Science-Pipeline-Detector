
# 시험환경 세팅 (코드 변경 X)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
X_train, X_test, y_train, y_test = exam_data_load(df, target='count')#, id_name='Id')

X_train.shape, X_test.shape, y_train.shape, y_test.shape

X_train.head()
X_test.head()

X_train.info()
X_test.info()

y_train.head()
y_test.head()

X_train.isna().sum(), X_test.isna().sum()

X_train.drop('datetime', axis = 1, inplace = True)
X_test.drop('datetime', axis = 1, inplace = True)
X_train.head(), X_test.head()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_col = ['temp', 'atemp', 'humidity', 'windspeed']
scaler.fit(X_train[num_col])
X_train[num_col] = scaler.transform(X_train[num_col])
X_test[num_col] = scaler.transform(X_test[num_col])
X_train.head(), X_test.head()

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
X_tr, X_ts, y_tr, y_ts = train_test_split(X_train, y_train['count'], test_size = 0.2, random_state = 504)

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

#params = {'n_estimators' : [80, 90, 100, 120], 'max_depth' : [5,6,7,8]}
#grid = GridSearchCV(xg, param_grid = params, scoring = 'neg_mean_squared_error', cv = 3)
#grid.fit(X_tr, y_tr)
#grid.best_score_, grid.best_params_

xg.fit(X_train, y_train['count'])
pred = xg.predict(X_test)
xg_mse = mean_squared_error(y_test['count'], pred)
np.sqrt(xg_mse)

sub = pd.DataFrame({'id': y_test['id'], 'count' : pred})
sub.to_csv('15945.csv', index = False)

pd.read_csv('./15945.csv')
