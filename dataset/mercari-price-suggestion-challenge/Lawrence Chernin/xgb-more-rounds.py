import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import *

def sp(key):
    return len(str(key).split())

train = pd.read_csv('../input/train.tsv', sep='\t')

mean_brand_name = train.groupby(['brand_name'], as_index=False)['price'].mean()
mean_shipping = train.groupby(['shipping'], as_index=False)['price'].mean()
mean_category_name = train.groupby(['category_name'], as_index=False)['price'].mean()
mean_item_condition_id = train.groupby(['item_condition_id'], as_index=False)['price'].mean()

test = pd.read_csv('../input/test.tsv', sep='\t')
df = pd.concat([train,test])

df = df.merge(mean_brand_name, on=['brand_name'], how='left', suffixes=('', 'mean_brand_name'))
df = df.merge(mean_shipping, on=['shipping'], how='left', suffixes=('', 'mean_shipping'))
df = df.merge(mean_category_name, on=['category_name'], how='left', suffixes=('', 'mean_category_name'))
df = df.merge(mean_item_condition_id, on=['item_condition_id'], how='left', suffixes=('', '_mean'))

df['category_name'] = pd.factorize(df['category_name'])[0]
df['brand_name'] = pd.factorize(df['brand_name'])[0]
df['item_description_w'] = df['item_description'].apply(sp)
df['item_description_l'] = df['item_description'].str.len()
df['item_description_s'] = df['item_description_l']/df['item_description_w']
df['name_w'] = df['name'].apply(sp)
df['name_l'] = df['name'].str.len()
df['name_s'] = df['name_l']/df['name_w']

col = [c for c in df.columns if c not in ['train_id', 'test_id', 'price', 'name', 'item_description']]

test_df = df[df['price'].isnull()]
df = df[~df['price'].isnull()]

x_train, x_valid, y_train, y_valid = model_selection.train_test_split(df[col], df['price'], test_size=0.25)

dtrain = xgb.DMatrix(x_train, y_train)
dvalid  = xgb.DMatrix(x_valid,  y_valid)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
params = {'min_child_weight': 20, 'eta': 0.05, 'colsample_bytree': 0.5, 'max_depth': 15,
            'subsample': 0.9, 'lambda': 2.0, 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear','tree_method': 'hist'}
model = xgb.train(params, dtrain, 1000, watchlist, verbose_eval=10, early_stopping_rounds=50)
test_df['price'] = model.predict(xgb.DMatrix(test_df[col]), ntree_limit=model.best_ntree_limit)
test_df['test_id'] = test_df['test_id'].astype(np.int)
test_df.loc[test_df['price'] < 0, 'price'] = 0
test_df[['test_id', 'price']].to_csv("sample.csv", index = False)