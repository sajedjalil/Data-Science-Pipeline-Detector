# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

df = pd.read_csv('../input/train.csv')
x_pred= pd.read_csv('../input/test.csv')


def convert_dates(x):
    x['date']=pd.to_datetime(x['date'])
    x['month']=x['date'].dt.month
    x['year']=x['date'].dt.year
    x['dayofweek']=x['date'].dt.dayofweek
    x.pop('date')
    return x

df = convert_dates(df)
x_pred = convert_dates(x_pred)

def add_avg(x):
    x['daily_avg']=x.groupby(['item','store','dayofweek'])['sales'].transform('mean')
    x['monthly_avg']=x.groupby(['item','store','month'])['sales'].transform('mean')
    return x
df = add_avg(df)

daily_avg = df.groupby(['item','store','dayofweek'])['sales'].mean().reset_index()
monthly_avg = df.groupby(['item','store','month'])['sales'].mean().reset_index()

def merge(x,y,col,col_name):
    x =pd.merge(x, y, how='left', on=None, left_on=col, right_on=col,
            left_index=False, right_index=False, sort=True,
             copy=True, indicator=False,validate=None)
    x=x.rename(columns={'sales':col_name})
    return x
x_pred=merge(x_pred, daily_avg,['item','store','dayofweek'],'daily_avg')
x_pred=merge(x_pred, monthly_avg,['item','store','month'],'monthly_avg')



x_train,x_test,y_train,y_test = train_test_split(df.drop('sales',axis=1),df.pop('sales'),random_state=123,test_size=0.2)

def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'mae'}
                    ,dtrain=matrix_train,num_boost_round=500, 
                    early_stopping_rounds=20,evals=[(matrix_test,'test')],)
    return model

model=XGBmodel(x_train,x_test,y_train,y_test)

submission = pd.DataFrame(x_pred.pop('id'))
y_pred = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)

submission['sales']= y_pred

submission.to_csv('sub3.csv',index=False)