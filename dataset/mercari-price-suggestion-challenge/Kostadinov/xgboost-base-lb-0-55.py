# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import xgboost as xgb
import numpy as np
import gc


# In[2]:


# price encode
encode_price = lambda x: np.log(1.0 + x) 
decode_price = lambda x: np.exp(x) - 1.0


# In[3]:


dtype={'brand_name':np.str, 'category_name':np.str, 'item_condition_id':np.int8, 'name':np.str, 'price':np.float32, 'shipping':np.int8}
train = pd.read_table('../input/train.tsv', dtype=dtype)
test = pd.read_table('../input/test.tsv', dtype=dtype)
print('train', len(train), 'test', len(test))

train['id'] = train['train_id'].values.astype(np.int32)
train['price'] = encode_price(train.price.values)
test['price'] = -1
test['id'] = test['test_id'].values.astype(np.int32)

train.drop(['train_id'], axis=1, inplace=True)
test.drop(['test_id'], axis=1, inplace=True)


data = pd.concat([train,test], ignore_index=True)
del train, test; gc.collect()


# In[4]:


# Category Split
category_index = ~data.category_name.isnull()
data['category_1'] = np.NaN
data.loc[category_index, 'category_1'] = LabelEncoder().fit_transform([cat.split('/')[0] for cat in data.category_name[category_index].values])
data['category_2'] = np.NaN
data.loc[category_index, 'category_2'] = LabelEncoder().fit_transform([cat.split('/')[1] for cat in data.category_name[category_index].values])
data['category_3'] = np.NaN
data.loc[category_index, 'category_3'] = LabelEncoder().fit_transform([cat.split('/')[2] for cat in data.category_name[category_index].values])


# In[5]:


# Brand
brand_index = ~data.brand_name.isnull()
data['brand'] = np.NaN
data.loc[brand_index, 'brand'] = LabelEncoder().fit_transform(data.brand_name[brand_index].values)


# In[6]:


data = data


# In[7]:


train = data[data.price != -1].copy()
test = data[data.price == -1].copy()
del data; gc.collect()
features = ['item_condition_id', 'shipping', 'brand', 'category_1', 'category_2', 'category_3']


# In[8]:


# Set up classifier
# Set up classifier
xgb_params= {  
            'eta': 0.7,
            'max_depth': 12,
            'objective':'reg:linear',
            'eval_metric':'rmse',
            'silent': 1
}


# In[9]:


kf = KFold(n_splits = 5, random_state = 1, shuffle = True)


# In[10]:


submission = pd.DataFrame()
submission['test_id'] = test.id
submission["price"] = 0


# In[12]:


for i, (train_index, valid_index) in enumerate(kf.split(train)):
    
    print('Fold', i)
    train_X, valid_X = train.loc[train_index, features], train.loc[train_index, features]
    train_y, valid_y = train.loc[train_index, 'price'], train.loc[train_index, 'price']
    
    train_matrix = xgb.DMatrix(train_X, train_y)
    validation_matrix = xgb.DMatrix(valid_X, valid_y)
    evallist  = [(validation_matrix,'validation')]
    
    model = xgb.train(xgb_params, train_matrix, 100,
                      evallist, verbose_eval=50)
    
    fold_prediction = model.predict(xgb.DMatrix(test[features]), ntree_limit=model.best_ntree_limit);
    submission["price"] += fold_prediction


# In[13]:


submission['price'] = decode_price(submission['price'] / 5.0)


# In[14]:


submission.to_csv("xgb_submission.csv", float_format='%.2f', index=False)