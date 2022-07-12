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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test_id = test.id
test = test.drop(['id'],axis=1)
print("训练集的规模：",train.shape)
print('测试集的大小：',test.shape)
print('训练集的特征：',train.columns)
print('测试集的特征：',test.columns)
print(train.sample(10))
print(test.sample(10))
print('------------------------------------------------------------------------')
print('训练集信息：',train.info())
print(train.describe())

print('------------------------------------------------------------------------')
print('测试集信息：',test.info())
print(test.describe())




#将时间序列转化为特征
train['date'] = pd.to_datetime(train['date'],infer_datetime_format=True)
train['month'] = train['date'].dt.month
train['day_of_week'] = train['date'].dt.dayofweek
train['year'] = train['date'].dt.year
train['week_of_year'] = train['date'].dt.weekofyear
train.set_index('date',inplace=True)

test['date'] = pd.to_datetime(test['date'],infer_datetime_format=True)
test['month'] = test['date'].dt.month
test['day_of_week'] = test['date'].dt.dayofweek
test['year'] = test['date'].dt.year
test['week_of_year'] = test['date'].dt.weekofyear
test.set_index('date',inplace=True)

print('训练集的列',train.columns)
print('测试集的列',test.columns)
print('------------------------------------------------------------------------')
print(train.sample(5))


#EDA探索性数据分析
'''sns.pairplot(train)
print("各大士多的平均售价：",train[['sales','store']].groupby(['store'],as_index=False).mean())
plt.figure(figsize=(12,10))
sns.stripplot(x=train.store,y=train.sales,data=train[['sales','store']].groupby(['store'],as_index=False).mean())
print('各商品的价格：',train[['item','sales']].groupby(['item'],as_index=False).mean())
plt.figure(figsize=(12,10))
sns.stripplot(x=train.item,y=train.sales,data=train[['item','sales']].groupby(['item'],as_index=False).mean())
print('各月份的平均价格',train[['month','sales']].groupby(['month'],as_index=False).mean())
plt.figure(figsize=(12,10))
sns.stripplot(x=train.month,y=train.sales,data=train[['month','sales']].groupby(['month'],as_index=False).mean())
print('一周内每天的平均价格：',train[['day_of_week','sales']].groupby(['day_of_week'],as_index=False).mean())
plt.figure(figsize=(12,10))
sns.boxplot(x=train.day_of_week,y=train.sales,data=train[['day_of_week','sales']].groupby(['day_of_week'],as_index=False).mean())
print('各年份的平均价格：',train[['year','sales']].groupby(['year'],as_index=False).mean())
plt.figure(figsize=(12,10))
sns.stripplot(x=train.year,y=train.sales,data=train[['year','sales']].groupby(['year'],as_index=False).mean())
print('一年内各星期的平均价格：',train[['week_of_year','sales']].groupby(['week_of_year'],as_index=False).mean())
plt.figure(figsize=(12,10))
sns.stripplot(x=train.week_of_year,y=train.sales,data=train[['week_of_year','sales']].groupby(['week_of_year'],as_index=False).mean())
'''

'''可以看到1，5，6，7商店卖出的价格要低一点,不是正态分布，而且偏离的厉害'''

#getX,Y
Y= train.sales
X = train.drop(['sales'],axis=1)


#标准化
'''X = (X - X.mean())/X.std
test = (test - test.mean())/test.std'''

model = RandomForestRegressor()
model.fit(X,Y)
cross_val = cross_val_score(model,X,Y,cv=5)
print("交叉验证的平均精准度：",cross_val.mean())

pred = model.predict(test)

tt = pd.DataFrame({'id':test_id,'sales':pred})
tt.to_csv('store items forecasting',index=False)