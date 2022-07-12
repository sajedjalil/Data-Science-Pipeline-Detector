# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import pandas as pd

df = pd.read_csv('../input/train.csv')
df.set_index('date', inplace = True)
df.index = pd.to_datetime(df.index)
#df.sort_index(inplace = True)
test = pd.read_csv('../input/test.csv')

test_df=test.copy()
test_df.set_index('date', inplace=True)
test_df.index = pd.to_datetime(test_df.index)
#test_df.sort_index(inplace = True)

ddf = pd.concat([df, test_df.drop('id', axis=1)])


data = ddf.groupby(['item', 'store'])
l=[g for g in data]


###############################################################################
import lightgbm as lgb


def thread_run(g):
    dd = g[1].copy()
    dd['dayofweek'] = dd.index.dayofweek
    dd['dayofmonth'] = dd.index.daysinmonth
    dd['quaterinyear'] = dd.index.quarter
    dd['year'] = dd.index.year
    dd['month'] = dd.index.month
    dd_main = dd.drop('sales', axis=1)
    y_main = dd.sales

    size = 90
    df_train, df_test, y_train, y_test = dd_main.iloc[:-size], dd_main.iloc[-size:],\
    y_main[:-size], y_main[-size:]
    
    train_data = lgb.Dataset(df_train.iloc[:-720], (y_train[:-720]))
    validation_data = lgb.Dataset(df_train.iloc[-720:], (y_train[-720:]))

    params = {
        'objective': 'regression',
        'num_leaves': 300,
        'subsample': 0.6143,
        'colsample_bytree': 0.6453,
        'min_split_gain': np.power(10, -2.5988),
        'reg_alpha': np.power(10, -2.2887),
        'reg_lambda': np.power(10, 1.7570),
        'min_child_weight': np.power(10, -0.1477),
        'verbose': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
        'max_depth': 5,
        'learning_rate': 0.01,
        'metric': 'mape',
        "feature_fraction":0.9,
        "bagging_freq": 6,
        "bagging_fraction":0.6 ,
        "task": "train"
    }

    print('Start training...')

    gbm = lgb.train(params,
                train_data,
                num_boost_round=15000,
                valid_sets=[train_data, validation_data],
                early_stopping_rounds=1000, verbose_eval=False)
    res = pd.Series(gbm.predict(df_test), index = df_test.index).round(0)
    return res


from sklearn.externals.joblib import Parallel, delayed
res_l = Parallel(n_jobs=8, backend="threading")(map(delayed(thread_run),l))


result = pd.DataFrame({'id': test.id.tolist(), 'sales': pd.concat(res_l).tolist()})
result.to_csv('output_lightgbm_split_joblib.csv', index =False)