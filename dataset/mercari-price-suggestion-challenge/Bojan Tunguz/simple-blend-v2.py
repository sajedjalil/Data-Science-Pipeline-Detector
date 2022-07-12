import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import *
import catboost as cboost
import csv


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

x_train, x_valid, y_train, y_valid = model_selection.train_test_split(df[col], df['price'], test_size=0.2)

dtrain = xgb.DMatrix(x_train, y_train)
dvalid  = xgb.DMatrix(x_valid,  y_valid)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
params = {}
params['max_depth'] = 7
params['seed'] = 99
params['tree_method'] = 'hist'
model = xgb.train(params, dtrain, 300, watchlist, verbose_eval=10, early_stopping_rounds=20)
XGB1_preds = model.predict(xgb.DMatrix(test_df[col]), ntree_limit=model.best_ntree_limit)

train = pd.read_csv('../input/train.tsv', sep='\t')
test = pd.read_csv('../input/test.tsv', sep='\t')
test['price'] = -1
df = pd.concat([train,test])
df['category_name'] = pd.factorize(df['category_name'])[0]
df['brand_name'] = pd.factorize(df['brand_name'])[0]
col = [c for c in df.columns if c not in ['train_id', 'test_id', 'price', 'name', 'item_description']]
test_df = df[df['price'] == -1]
df = df[df['price'] != -1]
x_train, x_valid, y_train, y_valid = model_selection.train_test_split(df[col], df['price'], test_size=0.2)

dtrain = xgb.DMatrix(x_train, y_train)
dvalid  = xgb.DMatrix(x_valid,  y_valid)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
params = {}
params['eta'] = 0.75
params['max_depth'] = 5
params['seed'] = 99
params['tree_method'] = 'hist'
model = xgb.train(params, dtrain, 500, watchlist, verbose_eval=10, early_stopping_rounds=20)
XGB2_preds = model.predict(xgb.DMatrix(test_df[col]), ntree_limit=model.best_ntree_limit)

df_train = pd.read_csv('../input/train.tsv', sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)
df_test = pd.read_csv('../input/test.tsv', sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)

categorical_features = ['item_condition_id', 'category_name', 'brand_name', 'shipping']

df_x_train = df_train[categorical_features].copy()
df_x_test = df_test[categorical_features].copy()
df_y = df_train['price']

# Factorize both train and test (avoid unseen categories in train)
def factorize(train, test, col):
    cat_ids = sorted(set(train[col].dropna().unique()) | set(test[col].dropna().unique()))

    cat_ids = {k:i for i, k in enumerate(cat_ids)}
    cat_ids[np.nan] = -1

    train[col] = train[col].map(cat_ids)
    test[col]  = test[col].map(cat_ids)

# Factorize string columns
factorize(df_x_train, df_x_test, 'category_name')
factorize(df_x_train, df_x_test, 'brand_name')

# Create train and test Pool of train
ptrain = cboost.Pool(df_x_train, df_y, cat_features=np.arange(len(categorical_features)),
                     column_description=categorical_features)

ptest = cboost.Pool(df_x_test, cat_features=np.arange(len(categorical_features)),
                     column_description=categorical_features)
                     
# Tune your parameters here!
cboost_params = {
    'nan_mode': 'Min',
    'loss_function': 'RMSE',  # Try 'LogLinQuantile' as well
    'iterations': 150,
    'learning_rate': 0.75,
    'depth': 5,
    'verbose': True
}

best_iter = cboost_params['iterations']  # Initial 'guess' it not using CV

model = cboost.CatBoostRegressor(**dict(cboost_params, verbose=False, iterations=best_iter))

fit_model = model.fit(ptrain)

CAT_preds= fit_model.predict(ptest)

median = df_train['price'].median()
train = df_train.groupby('category_name')['price'].median()
price_dict = train.to_dict()

MEDIAN_preds = []
for i, row in df_test.iterrows():
    category_name = row['category_name']
    if(category_name not in price_dict):
        MEDIAN_preds.append(median)
    else:
        MEDIAN_preds.append(price_dict[category_name])
        
MEDIAN_preds = np.array(MEDIAN_preds)

preds = np.clip(0.4*XGB1_preds + 0.35*CAT_preds+0.15*XGB2_preds+0.1*MEDIAN_preds, 0, 10000000000)
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['price'] = preds
sub.to_csv('blend_sub_2.csv', index=False)