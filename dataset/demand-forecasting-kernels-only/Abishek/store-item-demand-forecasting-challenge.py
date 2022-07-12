# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as pyplot
#from  sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/demand-forecasting-kernels-only/train.csv')
train.columns.tolist()
test = pd.read_csv('../input/demand-forecasting-kernels-only/test.csv')
test.columns.tolist()
train.shape
test.shape
features = train[['store','item']]
xgb = XGBRegressor(n_estimators=1000,learning_rate=0.05)
xgb.fit(train[['store','item']],train['sales'],early_stopping_rounds=5,eval_set=[(train[['store','item']],train['sales'])],verbose=False)
#rf.fit(X = features,y = train['sales'])
test['sales'] = xgb.predict(test[['store','item']])
my_submission = pd.DataFrame({'ID':test.id,'sales':test['sales'] })
my_submission.to_csv('submission.csv',index=False)
