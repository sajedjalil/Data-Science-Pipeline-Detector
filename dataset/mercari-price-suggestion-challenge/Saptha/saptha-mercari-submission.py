# coding: utf-8

# In[34]:


import pandas as pd


# In[35]:


import numpy as np


# In[36]:


import xgboost as xg

import math

from scipy.stats import mode


# In[37]:


df_train = pd.read_csv('../input/train.tsv', sep='\t')


# In[38]:


df_train = df_train[df_train['price'] != 0]


# In[39]:


df_test = pd.read_csv('../input/test.tsv', sep='\t')


# In[40]:


df_train['category_name'] = df_train['category_name'].fillna('None')
df_train['brand_name'] = df_train['brand_name'].fillna('None')
df_train['item_description'] = df_train['item_description'].fillna('No description yet')


# In[41]:


df_test['category_name'] = df_test['category_name'].fillna('None')
df_test['brand_name'] = df_test['brand_name'].fillna('None')
df_test['item_description'] = df_test['item_description'].fillna('No description yet')


# In[42]:

raw_data = pd.concat([df_train, df_test], 0)
nrow_train = df_train.shape[0]


# In[12]:


raw_data[['cat1','cat2','cat3','cat4','cat5']] = raw_data['category_name'].str.split('/', expand=True).fillna('None')


# In[13]:


#raw_data = pd.concat([raw_data, pd.get_dummies(raw_data['cat1'], prefix='cat1')], axis=1)
raw_data = pd.concat([raw_data, pd.get_dummies(raw_data['item_condition_id'], prefix='item_cond')], axis=1)


# In[49]:


print(len(raw_data.index))




# In[51]:


print(raw_data.head())


# In[53]:


raw_data.columns = raw_data.columns.str.replace('&', '')
raw_data.columns = raw_data.columns.str.replace(' ', '_')
print(raw_data.columns)

# In[55]:


from sklearn.preprocessing import LabelBinarizer


# In[56]:


from sklearn.preprocessing import LabelEncoder


# In[57]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[58]:


from sklearn_pandas import gen_features


# In[59]:


from sklearn.preprocessing import StandardScaler


# In[60]:


from sklearn_pandas import DataFrameMapper, cross_val_score


# In[61]:


#Add the features of the dataframe that you want to transform and/or combine
mapper = DataFrameMapper([
     ('item_description', TfidfVectorizer(max_df=0.5, min_df=5, ngram_range = (2,3), sublinear_tf=True, use_idf=True, stop_words='english')),
     ('name', TfidfVectorizer(max_df=0.5, min_df=5, ngram_range = (1,3), sublinear_tf=True, use_idf=True, stop_words='english')),
     ('shipping', LabelEncoder()),
     ('item_cond_1', LabelEncoder()),
     ('item_cond_2', LabelEncoder()),
     ('item_cond_3', LabelEncoder()),
     ('item_cond_4', LabelEncoder()),
     ('item_cond_5', LabelEncoder()),
     ('cat2', LabelBinarizer(sparse_output=True)),
     ('cat1', LabelBinarizer(sparse_output=True)),
     ('cat3', LabelBinarizer(sparse_output=True)),
     ('cat4', LabelBinarizer(sparse_output=True)),
     ('cat5', LabelBinarizer(sparse_output=True)),
     ('brand_name', LabelBinarizer(sparse_output=True))
 ], sparse=True)




train_test = mapper.fit_transform(raw_data)


# In[65]:


trainX = train_test[:nrow_train]


# In[86]:


#print(trainX_nb)


# In[87]:


#trainX_b


# In[67]:


testX = train_test[nrow_train:]


# In[90]:


#testX_nb


# In[91]:


#testX_b


# In[69]:


train_data = raw_data[:nrow_train]


# In[71]:


trainY = np.log1p(train_data['price'])


# In[79]:


params = {
'obj':'reg:linear',
'booster':'gblinear',
'eval_metric':'rmse',
'lambda':2.75, 
'lambda_bias':5000, 
'alpha':0}
num_rounds = 250


# In[80]:


xgtrainX = xg.DMatrix(trainX, label=trainY)


# In[82]:


watchlist = [(xgtrainX, 'train')]


# In[83]:


params['eval_metric'] = 'rmse'


# In[84]:


dmatrix = xg.train(params,
                     xgtrainX,
                     num_rounds, watchlist, early_stopping_rounds=10)


# In[88]:



print("Best train score: {}".format(dmatrix.best_score))
print("Best train iteration: {}".format(dmatrix.best_iteration))
print("Best train number of trees limit : {}".format(dmatrix.best_ntree_limit))


# In[89]:


#Repeat the same steps for test data and use the below commands 
testY = dmatrix.predict(xg.DMatrix(testX))

# In[98]:


#f=lambda a: (abs(a)+a)/2
f=lambda a: np.ceil(abs(a)*2)/2

# In[99]:


#res_nb = pd.DataFrame({"test_id": df_test_nb['test_id'], "price": np.expm1(testY_nb) })
res = pd.DataFrame({"test_id": df_test['test_id'], "price": np.expm1(testY), "category_name": df_test['category_name'] })

# In[101]:
#group_cat = df_train.groupby(['category_name'])['price'].apply(lambda x: np.mean(x))
group_cat = df_train.groupby(['category_name'])['price'].apply(lambda x: x.mode()[0])
for index, row in res.iterrows():
    if(row['price'] < 0):
        filt = '^' + row['category_name'] + '$'
        temp_val = group_cat.filter(regex=filt, axis=0)
        if(temp_val.empty):
            row['mode_price'] = 0
            res.loc[index, 'price'] = row['mode_price']
        else:
            row['mode_price'] = temp_val.values[0]
            res.loc[index, 'price'] = row['mode_price']
        print(row['test_id'], row['category_name'], row['price'], row['mode_price'])
    elif(row['price'] > 0 and row['price'] < 3):
        row['mode_price'] = 3
        res.loc[index, 'price'] = row['mode_price']
        #print(row['test_id'], row['category_name'], row['price'], row['mode_price'])



result = res.sort_values(['test_id'])


# In[103]:


result = np.round(result, decimals=3)


# In[104]:
result1 = pd.DataFrame({"test_id": result['test_id'], "price": f(result['price']) })

result1[["test_id", "price"]].to_csv("submission.csv", index = False, float_format='%.3f')


# In[105]:


result.head()