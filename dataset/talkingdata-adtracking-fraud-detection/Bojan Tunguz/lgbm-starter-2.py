import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc

path = '../input/'

def dataPreProcessTime(df):
    df['click_time'] = pd.to_datetime(df['click_time']).dt.date
    df['click_time'] = df['click_time'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    
    return df

start_time = time.time()

train = pd.read_csv(path+"train.csv", skiprows=130000000, nrows=70000000)
train.columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']


print('[{}] Finished loading data'.format(time.time() - start_time))

train = dataPreProcessTime(train)


y = train['is_attributed']
train.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)


print('[{}] Starting LGBM Training'.format(time.time() - start_time))

params = {
    'num_leaves': 31,
    'objective': 'binary',
    'min_data_in_leaf': 200,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.85,
    'bagging_freq': 3,
    'metric': 'auc',
    'num_threads': 4,
    'scale_pos_weight':400
}

MAX_ROUNDS = 650



x1, x2, y1, y2 = train_test_split(train, y, test_size=0.1, random_state=99)

del train, y
gc.collect()

dtrain = lgb.Dataset(x1, label=y1)
dval = lgb.Dataset(x2, label=y2, reference=dtrain)

del x1, x2, y1, y2
gc.collect()

model = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=10)


del dtrain, dval
gc.collect()

print('[{}] Finished LGBM Training'.format(time.time() - start_time))

test = pd.read_csv(path+"test.csv")
test = dataPreProcessTime(test)
sub = pd.DataFrame()
sub['click_id'] = test['click_id']
test.drop('click_id', axis=1, inplace=True)
sub['is_attributed'] = model.predict(test, num_iteration=model.best_iteration or MAX_ROUNDS)
sub.to_csv('lgb_sub.csv',index=False)