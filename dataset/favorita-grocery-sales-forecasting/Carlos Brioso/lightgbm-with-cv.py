# Data preparation is based on https://www.kaggle.com/lscoelho/four-regressors-nn-bayridge-bag-and-xgb/code

import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model, metrics
import gc; gc.enable()
import random
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import TheilSenRegressor, BayesianRidge
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_log_error

import time

np.random.seed(1343)

# store the total processing time
start_time = time.time()
tcurrent   = start_time

# read datasets
dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8', 'onpromotion':str}
data = {
    'tra': pd.read_csv('../input/train.csv', dtype=dtypes, parse_dates=['date']),
    'tes': pd.read_csv('../input/test.csv', dtype=dtypes, parse_dates=['date']),
    'ite': pd.read_csv('../input/items.csv'),
    'sto': pd.read_csv('../input/stores.csv'),
    'trn': pd.read_csv('../input/transactions.csv', parse_dates=['date']),
    'hol': pd.read_csv('../input/holidays_events.csv', dtype={'transferred':str}, parse_dates=['date']),
    'oil': pd.read_csv('../input/oil.csv', parse_dates=['date']),
    }

# dataset processing
print('Datasets processing')

train = data['tra'][data['tra']['date'].dt.year >= 2016]
del data['tra']; gc.collect();
target = train['unit_sales'].values
target[target < 0.] = 0.
train['unit_sales'] = np.log1p(target)

#test.head()

# train.tail()
def df_lbl_enc(df):
    for c in df.columns:
        if df[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            df[c] = lbl.fit_transform(df[c])
            print(c)
    return df

def df_transform(df):
    df['date'] = pd.to_datetime(df['date'])
    df['yea'] = df['date'].dt.year
    df['mon'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['date'] = df['date'].dt.dayofweek
    df['onpromotion'] = df['onpromotion'].map({'False': 0, 'True': 1})
    df['perishable'] = df['perishable'].map({0:1.0, 1:1.25})
    df = df.fillna(-1)
    return df

data['ite'] = df_lbl_enc(data['ite'])
train = pd.merge(train, data['ite'], how='left', on=['item_nbr'])
test = pd.merge(data['tes'], data['ite'], how='left', on=['item_nbr'])
del data['tes']; gc.collect();
del data['ite']; gc.collect();

train = pd.merge(train, data['trn'], how='left', on=['date','store_nbr'])
test = pd.merge(test, data['trn'], how='left', on=['date','store_nbr'])
del data['trn']; gc.collect();
target = train['transactions'].values
target[target < 0.] = 0.
train['transactions'] = np.log1p(target)

data['sto'] = df_lbl_enc(data['sto'])
train = pd.merge(train, data['sto'], how='left', on=['store_nbr'])
test = pd.merge(test, data['sto'], how='left', on=['store_nbr'])
del data['sto']; gc.collect();

data['hol'] = data['hol'][data['hol']['locale'] == 'National'][['date','transferred']]
data['hol']['transferred'] = data['hol']['transferred'].map({'False': 0, 'True': 1})
train = pd.merge(train, data['hol'], how='left', on=['date'])
test = pd.merge(test, data['hol'], how='left', on=['date'])
del data['hol']; gc.collect();

train = pd.merge(train, data['oil'], how='left', on=['date'])
test = pd.merge(test, data['oil'], how='left', on=['date'])
del data['oil']; gc.collect();

train = df_transform(train)
test = df_transform(test)
col = [c for c in train if c not in ['id', 'date', 'unit_sales']]
#test.shape[0]
#test.head()

x1 = train[(train['yea'] != 2016) & (train['mon'] != 8)][col]
x2 = train[(train['yea'] == 2016) & (train['mon'] == 8)][col]
#x1.head()

y1 = train[(train['yea'] != 2016) & (train['mon'] != 8)]['unit_sales'].values
y2 = train[(train['yea'] == 2016) & (train['mon'] == 8)]['unit_sales'].values
del train; gc.collect();


def NWRMSLE(preds, train_data):
    return 'nwrmsle', mean_squared_log_error(train_data.get_label(), preds), False


import lightgbm as lgb
# create dataset for lightgbm
lgb_train = lgb.Dataset(x1, y1)
lgb_eval = lgb.Dataset(x2, y2, reference=lgb_train)

mean_squared_log_error(np.exp(lgb_train.get_label()), np.exp(lgb_train.get_label()))

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'poisson',
    'metric': 'NWRMSLE',
    'num_leaves': 50,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Start training...')
# train
evals_result = {} # to record eval results for plotting
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets= (lgb_train, lgb_eval),
                verbose_eval=50,
                feval=NWRMSLE, 
                evals_result=evals_result,
                early_stopping_rounds=20)

print('Save model...')
#fig, ax = plt.subplots()
#ax = lgb.plot_metric(evals_result, metric='l1')
#plt.show()

# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm.predict(test[col].values, num_iteration=gbm.best_iteration)
y_pred[0:5]
# eval
sub = pd.DataFrame(test['id'])
sub['unit_sales'] = 0
sub['unit_sales'] = y_pred
#sub.head(50)
sub.loc[sub['unit_sales'] < 0,['unit_sales']]= 0


sub.to_csv('subm01.csv.gz', index=False,
float_format='%.3f', compression='gzip')