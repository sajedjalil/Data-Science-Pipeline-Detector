# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import gc
import sys

dtypes = {
        'ip':'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8'
        }

#tr_org=pd.read_csv('../input/train.csv',dtype=dtypes, usecols=[0,1,2,3,4,5,7])
train=pd.read_csv('../input/train.csv',dtype=dtypes, usecols=[0,1,2,3,4,7]) #skip attributed_time
test=pd.read_csv('../input/test.csv',dtype=dtypes, usecols=[0,1,2,3,4,5])
print('p1')

import lightgbm as lgb

params = {
            'learning_rate':0.03,
            'max_depth':20,
            'scale_pos_weight':400,
            'num_leaves':30, 
            'num_trees':500, 
            'objective':'binary', 
            'lambda_l2':100,
            'metric':'auc'
}

num_round =300

xg_tr = lgb.Dataset(train.drop('is_attributed',axis=1).values, label=train['is_attributed'].values)
bst=lgb.train(params, xg_tr, num_round)
pred=bst.predict(test.drop('click_id',axis=1).values)

del train, xg_tr
gc.collect()
print('p2')

res=pd.DataFrame()
res['click_id']=test['click_id']
del test
res['is_attributed']=pred
res.to_csv('LGB.csv',index=False)