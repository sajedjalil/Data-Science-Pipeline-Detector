#Data processing part reference- https://www.kaggle.com/the1owl/playing-in-the-sandbox
import numpy as np
import pandas as pd
from sklearn import *
import nltk, datetime
import h2o
from h2o.automl import H2OAutoML
h2o.init()

train = pd.read_csv('../input/sales_train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')
items = pd.read_csv('../input/items.csv')
item_cats = pd.read_csv('../input/item_categories.csv')
shops = pd.read_csv('../input/shops.csv')
print('train:', train.shape, 'test:', test.shape)

#Column difference betwee train and test
[c for c in train.columns if c not in test.columns]

train.head()
test.head()

#Adding Features

#Text Features
feature_cnt = 25
tfidf = feature_extraction.text.TfidfVectorizer(max_features=feature_cnt)
items['item_name_len'] = items['item_name'].map(len) #Lenth of Item Description
items['item_name_wc'] = items['item_name'].map(lambda x: len(str(x).split(' '))) #Item Description Word Count
txtFeatures = pd.DataFrame(tfidf.fit_transform(items['item_name']).toarray())
cols = txtFeatures.columns
for i in range(feature_cnt):
    items['item_name_tfidf_' + str(i)] = txtFeatures[cols[i]]
items.head()

#Text Features
feature_cnt = 25
tfidf = feature_extraction.text.TfidfVectorizer(max_features=feature_cnt)
item_cats['item_category_name_len'] = item_cats['item_category_name'].map(len)  #Lenth of Item Category Description
item_cats['item_category_name_wc'] = item_cats['item_category_name'].map(lambda x: len(str(x).split(' '))) #Item Category Description Word Count
txtFeatures = pd.DataFrame(tfidf.fit_transform(item_cats['item_category_name']).toarray())
cols = txtFeatures.columns
for i in range(feature_cnt):
    item_cats['item_category_name_tfidf_' + str(i)] = txtFeatures[cols[i]]
item_cats.head()

#Text Features
feature_cnt = 25
tfidf = feature_extraction.text.TfidfVectorizer(max_features=feature_cnt)
shops['shop_name_len'] = shops['shop_name'].map(len)  #Lenth of Shop Name
shops['shop_name_wc'] = shops['shop_name'].map(lambda x: len(str(x).split(' '))) #Shop Name Word Count
txtFeatures = pd.DataFrame(tfidf.fit_transform(shops['shop_name']).toarray())
cols = txtFeatures.columns
for i in range(feature_cnt):
    shops['shop_name_tfidf_' + str(i)] = txtFeatures[cols[i]]
shops.head()

#Make Monthly
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
train = train.drop(['date','item_price'], axis=1)
train = train.groupby([c for c in train.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
train = train.rename(columns={'item_cnt_day':'item_cnt_month'})
#Monthly Mean
shop_item_monthly_mean = train[['shop_id','item_id','item_cnt_month']].groupby(['shop_id','item_id'], as_index=False)[['item_cnt_month']].mean()
shop_item_monthly_mean = shop_item_monthly_mean.rename(columns={'item_cnt_month':'item_cnt_month_mean'})
#Add Mean Feature
train = pd.merge(train, shop_item_monthly_mean, how='left', on=['shop_id','item_id'])
#Last Month (Oct 2015)
shop_item_prev_month = train[train['date_block_num']==33][['shop_id','item_id','item_cnt_month']]
shop_item_prev_month = shop_item_prev_month.rename(columns={'item_cnt_month':'item_cnt_prev_month'})
shop_item_prev_month.head()
#Add Previous Month Feature
train = pd.merge(train, shop_item_prev_month, how='left', on=['shop_id','item_id']).fillna(0.)
#Items features
train = pd.merge(train, items, how='left', on='item_id')
#Item Category features
train = pd.merge(train, item_cats, how='left', on='item_category_id')
#Shops features
train = pd.merge(train, shops, how='left', on='shop_id')
train.head()

test['month'] = 11
test['year'] = 2015
test['date_block_num'] = 34
#Add Mean Feature
test = pd.merge(test, shop_item_monthly_mean, how='left', on=['shop_id','item_id']).fillna(0.)
#Add Previous Month Feature
test = pd.merge(test, shop_item_prev_month, how='left', on=['shop_id','item_id']).fillna(0.)
#Items features
test = pd.merge(test, items, how='left', on='item_id')
#Item Category features
test = pd.merge(test, item_cats, how='left', on='item_category_id')
#Shops features
test = pd.merge(test, shops, how='left', on='shop_id')
test['item_cnt_month'] = 0.
test.head()

#Label Encoding
for c in ['shop_name','item_name','item_category_name']:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[c].unique())+list(test[c].unique()))
    train[c] = lbl.fit_transform(train[c].astype(str))
    test[c] = lbl.fit_transform(test[c].astype(str))
    print(c)

col = [c for c in train.columns if c not in ['item_cnt_month']]
#Validation Hold Out Month
train_copy = train.copy()
test_copy = test.copy()
#x1 = train[train['date_block_num']<33]
#y1 = np.log1p(x1['item_cnt_month'].clip(0.,20.))
train['item_cnt_month'] = np.log1p(train['item_cnt_month'].clip(0.,20.))
#x1 = x1[col]
#x2 = train[train['date_block_num']==33]
#y2 = np.log1p(x2['item_cnt_month'].clip(0.,20.))
#x2 = x2[col]
print('Pre-processing done!')
#x1.isnull().values.any()
test = test.drop('ID', axis=1)

htrain = h2o.H2OFrame(train)
htest = h2o.H2OFrame(test)

#htrain.drop(['item_cnt_month'])
#htest.drop(['ID'])

x =htrain.columns
y ='item_cnt_month'
x.remove(y)

def RMSLE(y_, pred):
    return metrics.mean_squared_error(y_, pred)**0.5

print('Starting h2o autoML model!')  

aml = H2OAutoML(max_runtime_secs = 3600)
aml.train(x=x, y =y, training_frame=htrain, leaderboard_frame = htest)

print('Generate predictions...')
htrain.drop(['item_cnt_month'])
preds = aml.leader.predict(htrain)
preds = preds.as_data_frame()
print('RMSLE h2o automl leader: ', RMSLE(train['item_cnt_month'].clip(0.,20.), preds))

preds = aml.leader.predict(htest)
preds = preds.as_data_frame()
test_copy['item_cnt_month'] = preds
test_copy['item_cnt_month'] = np.expm1(test['item_cnt_month']).clip(0.,20.)
test_copy[['ID','item_cnt_month']].to_csv('submission-manoj_h2o.csv', index=False)
print('Done, Cheers!!')

