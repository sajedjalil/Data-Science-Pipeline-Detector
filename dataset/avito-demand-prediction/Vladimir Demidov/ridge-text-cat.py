import numpy as np
import pandas as pd
import scipy
from math import sqrt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from nltk.corpus import stopwords
stopWords = stopwords.words('russian')


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
subm = pd.read_csv('../input/sample_submission.csv')


y_train = train['deal_probability']
nrow_train = train.shape[0]
df = pd.concat([train, test], axis=0)
df = df.drop(['item_id', 'image', 'price', 'image_top_1', 'item_seq_number'], axis=1)

df.activation_date = pd.to_datetime(df.activation_date)
df['day_of_week'] = df.activation_date.apply(lambda x: x.weekday())

pop_user = df['user_id'].value_counts().index[:5000]
df.loc[~df['user_id'].isin(pop_user), 'user_id'] = 'Other'


print('encoding...')
df['param_1'] = df['param_1'].fillna(' ')
df['param_2'] = df['param_2'].fillna(' ')
df['param_3'] = df['param_3'].fillna(' ')

label_bin = LabelBinarizer(sparse_output=True)

user_id = label_bin.fit_transform(df['user_id'].values)
city = label_bin.fit_transform(df['city'].values)
region = label_bin.fit_transform(df['region'].values)    
category_name = label_bin.fit_transform(df['category_name'].values)
parent_category_name = label_bin.fit_transform(df['parent_category_name'].values)
user_type = label_bin.fit_transform(df['user_type'].values)    
param_1 = label_bin.fit_transform(df['param_1'].values)    
param_2 = label_bin.fit_transform(df['param_2'].values)    
param_3 = label_bin.fit_transform(df['param_3'].values) 


print('counting...')
df['description'] = df['description'].fillna(' ')

count = CountVectorizer()
title = count.fit_transform(df['title'])

tfidf = TfidfVectorizer(max_features=50000, stop_words = stopWords)
description = tfidf.fit_transform(df['description'])


print('dummies...')
dummies = scipy.sparse.csr_matrix(pd.get_dummies(df[[
                'day_of_week']].astype(float),
                 sparse = True).values)


X = scipy.sparse.hstack((user_id,
                         city,
                         region,
                         category_name,
                         parent_category_name,
                         user_type,
                         param_1,
                         param_2,
                         param_3,
                         title,
                         description,
                         dummies)).tocsr()
print(X.shape)

X_train = X[:nrow_train]

model = Ridge()
print('fitting...')
model.fit(X_train, y_train)

y_pred = model.predict(X_train)
print(sqrt(mean_squared_error(y_train, y_pred)))

X_test = X[nrow_train:]
preds = model.predict(X_test)

subm['deal_probability'] = preds
subm['deal_probability'].clip(0.0, 1.0, inplace=True)
subm.to_csv('submission.csv', index=False)