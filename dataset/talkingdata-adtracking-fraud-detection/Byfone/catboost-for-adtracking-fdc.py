import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostClassifier
import gc
dtypes = {
        'ip':'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8'
        }
dnames = ['ip','app','device','os','channel','click_time','attributed_time','is_attributed']
eco_train = pd.read_csv('../input/train.csv', dtype=dtypes, names=dnames, skiprows=int(1e8), nrows=3e6)
del eco_train['attributed_time']
print(eco_train.describe())
gc.collect()
eco_train['day'] = pd.to_datetime(eco_train.click_time).dt.day.astype('uint8')
eco_train['hour'] = pd.to_datetime(eco_train.click_time).dt.hour.astype('uint8')
eco_train['wday']  = pd.to_datetime(eco_train.click_time).dt.dayofweek.astype('uint8')
del eco_train['click_time']
print('eco_train loaded.', eco_train.columns)
cf = [0,1,2,3,4]
m = CatBoostClassifier(eval_metric='AUC')
m.fit(eco_train.drop('is_attributed', axis=1), eco_train['is_attributed'], cat_features=cf)
print('fitted.')
del eco_train
gc.collect()
eco_test = pd.read_csv('../input/test.csv', dtype=dtypes, usecols=['click_id','ip','app','device','os','channel','click_time'])
eco_test['day'] = pd.to_datetime(eco_test.click_time).dt.day.astype('uint8')
eco_test['hour'] = pd.to_datetime(eco_test.click_time).dt.hour.astype('uint8')
eco_test['wday']  = pd.to_datetime(eco_test.click_time).dt.dayofweek.astype('uint8')
del eco_test['click_time']
print('eco_test loaded.', eco_test.columns)
sub = pd.DataFrame(columns=['click_id', 'is_attributed'])
sub['click_id'] = eco_test['click_id'].astype('uint32')
sub['is_attributed'] = m.predict_proba(eco_test.drop('click_id', axis=1))
print('predicted.')
sub['is_attributed'] = sub['is_attributed'].apply(lambda x: 1-x)
print('transformed.')
sub.to_csv('cb_submit.csv', index=None)