#regurgitation of other kernels

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
pd.set_option('display.float_format', lambda x: '%.5f' % x)
from scipy.cluster import hierarchy as hc
import os
import numpy as np

types_dict_train = {'train_id': 'int64',
             'item_condition_id': 'int8',
             'price': 'float64',
             'shipping': 'int8'}
train = pd.read_csv('../input/train.tsv',delimiter='\t',low_memory=True,dtype=types_dict_train)
types_dict_test = {'test_id': 'int64',
             'item_condition_id': 'int8',
             'shipping': 'int8'}
test = pd.read_csv('../input/test.tsv',delimiter='\t',low_memory= True,dtype=types_dict_test)

def display_all(df):
    with pd.option_context("display.max_rows", 1000): 
        with pd.option_context("display.max_columns", 1000): 
            display(df)

train.category_name = train.category_name.astype('category')
train.item_description = train.item_description.astype('category')

train.name = train.name.astype('category')
train.brand_name = train.brand_name.astype('category')
test.category_name = test.category_name.astype('category')
test.item_description = test.item_description.astype('category')

test.name = test.name.astype('category')
test.brand_name = test.brand_name.astype('category')


train.apply(lambda x: x.nunique())

test.apply(lambda x: x.nunique())

train.isnull().sum(),train.isnull().sum()/train.shape[0]

test.isnull().sum(),test.isnull().sum()/test.shape[0]

os.makedirs('data/tmp',exist_ok=True)

train = train.rename(columns = {'train_id':'id'})
print("TRAIN")
print(train.head())
test = test.rename(columns = {'test_id':'id'})

train['is_train'] = 1
test['is_train'] = 0
train_test_combine = pd.concat([train.drop(['price'],axis =1),test],axis = 0)
train_test_combine.category_name = train_test_combine.category_name.astype('category')
train_test_combine.item_description = train_test_combine.item_description.astype('category')

train_test_combine.name = train_test_combine.name.astype('category')
train_test_combine.brand_name = train_test_combine.brand_name.astype('category')
train_test_combine.name = train_test_combine.name.cat.codes
train_test_combine.category_name = train_test_combine.category_name.cat.codes
train_test_combine.brand_name = train_test_combine.brand_name.cat.codes
train_test_combine.item_description = train_test_combine.item_description.cat.codes

df_test = train_test_combine.loc[train_test_combine['is_train']==0]
df_train = train_test_combine.loc[train_test_combine['is_train']==1]
df_test = df_test.drop(['is_train'],axis=1)
df_train = df_train.drop(['is_train'],axis=1)

df_train['price'] = train.price
df_train['price'] = df_train['price'].apply(lambda x: np.log(x) if x>0 else x)

x_train,y_train = df_train.drop(['price'],axis =1),df_train.price

m = RandomForestRegressor(n_jobs=-1,min_samples_leaf=50,n_estimators=20)
m.fit(x_train, y_train)
m.score(x_train,y_train)

preds = m.predict(df_test)
preds = pd.Series(np.exp(preds))


submit = pd.concat([df_test.id,preds],axis=1)
submit.columns = ['test_id','price']
submit.to_csv('data/submit_rf_base.csv',index=False)