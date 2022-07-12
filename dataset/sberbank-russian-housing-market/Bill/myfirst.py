# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from random import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

id_test = test.id

y_train = train["price_doc"] * .9685 + 10.
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

#can't merge train with test because the kernel run for very long time

delete   = ['sub_area','ecology','material']
this     = x_train.select_dtypes(include = ['object']).columns.values.tolist()
this.extend(delete)


x_train = pd.concat([x_train,y_train],axis=1)

x_train['price_pre_sq'] = x_train['price_doc']/(x_train['full_sq']+0.01)

for i in this:
    x_train[i+'_new']   = x_train.groupby([i])['price_pre_sq'].transform('mean')
    x_test[i+'_new']    = 0

y                        = x_train['price_pre_sq']
y_train                  = x_train['price_doc']
x_train.drop(['price_doc','price_pre_sq'],axis=1,inplace=True)
df_all                   = pd.concat([x_train,x_test],axis=0)

for i in this:
    df_all[i+'_new']     = df_all.groupby([i])[i+'_new'].transform('max')


id_test                  = df_all.iloc[len(y_train):,0]
df_all.drop(this,axis=1,inplace=True)


df_all.index        = range(len(df_all))
y_train.index       = range(len(y_train))

x_train = df_all.iloc[:len(y_train),:]
x_test  = df_all.iloc[len(y_train):,:]


xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round= num_boost_rounds)


y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

output.to_csv('xgbSub.csv', index=False)

