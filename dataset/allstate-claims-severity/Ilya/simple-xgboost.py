

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# load data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# encoding categorical variables
for feature in df_train.columns:
    if 'cat' in feature:
        gr_feature = df_train.groupby(df_train[feature])['loss'].mean()
        df_train[feature] = df_train[feature].map(gr_feature)
        df_test[feature] = df_test[feature].map(gr_feature)


        
# split data        
y_train = np.log(df_train['loss'] + 200)
x_train = df_train.drop(['loss', 'id'], axis = 1)
x_test = df_test.drop(['id'], axis = 1)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, 
                                                      test_size = 0.15, 
                                                      random_state = 345)

# apply xgboost
d_train = xgb.DMatrix(x_train, y_train)
d_valid = xgb.DMatrix(x_valid, y_valid)
d_test = xgb.DMatrix(x_test)

xgb_params = {
    'learning_rate'   : 0.02,
    'max_depth'       : 7,
    'min_child_weight': 1,
    'subsample'       : 0.7,
    'colsample_bytree': 0.7,

    'objective': 'reg:linear',
    'silent'   : 1,
    'seed'     : 120
}

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))
    
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

clf = xgb.train(xgb_params, d_train, 2500, watchlist, early_stopping_rounds=30, 
                verbose_eval=20, feval=xg_eval_mae, maximize=False)

p_test = np.exp(clf.predict(d_test)) - 200

sub = pd.DataFrame()
sub['id'] = df_test['id']
sub['loss'] = p_test
sub.to_csv('xqboost_start2.csv', index=False)