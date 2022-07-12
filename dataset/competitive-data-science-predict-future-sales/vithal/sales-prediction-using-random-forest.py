# -*- coding: utf-8 -*-

#Created on Wed May 13 17:26:31 2020

#@author: Vithal Nistala

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor


items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
cat = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
test  = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
#print(submission.head())
#print(train.isnull().sum())

grp = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day':['sum']})
x= np.array(list(map(list, grp.index.values)))
y_train = grp.values
test['date_block_num'] = train['date_block_num'].max()+1
x_test = test[['date_block_num', 'shop_id', 'item_id']].values
oh1 = OneHotEncoder(categories='auto').fit(x[:,1].reshape(-1, 1))
x1 = oh1.transform(x[:,1].reshape(-1, 1))
x1_t = oh1.transform(x_test[:,1].reshape(-1, 1))
x_train= np.concatenate((x[:,:1],x1.toarray(),x[:,2:]),axis=1)
x_test = np.concatenate((x_test[:,:1],x1_t.toarray(),x_test[:,2:]),axis=1)


rfr = RandomForestRegressor()
rfr.fit(x_train,y_train.ravel())
y_test = rfr.predict(x_test)
submission['item_cnt_month'] = y_test
submission.to_csv('submission.csv',index=False)


