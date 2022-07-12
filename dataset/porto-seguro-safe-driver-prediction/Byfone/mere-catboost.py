import numpy as np
import pandas as pd
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
ss = pd.read_csv('../input/sample_submission.csv')

from catboost import Pool, CatBoostRegressor
m = CatBoostRegressor()
train_pool = Pool(train.drop(['id', 'target'], axis=1), train['target'])
test_pool = Pool(test.drop(['id'], axis=1))
m.fit(train_pool)
ss['target'] = m.predict(test_pool).clip(0.0, 1.0)
ss.to_csv('subm.csv', index=None)