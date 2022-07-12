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
import seaborn as sns
from scipy.stats import norm, skew
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split
import math


train_df = pd.read_table('../input/train.tsv')
test_df = pd.read_table('../input/test_stg2.tsv')
y = train_df['price']

test_id = test_df['test_id']

print(train_df.head())


train_df.drop(['train_id','price'],axis=1,inplace=True)
test_df.drop(['test_id'],axis=1,inplace=True)
print(train_df.shape)
print(test_df.shape)

name_null_count = train_df['name'].isnull().sum()
item_condition_id_null_count = train_df['item_condition_id'].isnull().sum()
category_name_null_count = train_df['category_name'].isnull().sum()
brand_name_null_count = train_df['brand_name'].isnull().sum()
shipping_null_count = train_df['shipping'].isnull().sum()
item_description_null_count = train_df['item_description'].isnull().sum()

#want to see null counts of every column
print(train_df.columns)
print('name_null_count',item_condition_id_null_count)
print('item_condition_id_null_count',item_condition_id_null_count)
print('category_name_null_count',category_name_null_count)
print('brand_name_null_count',brand_name_null_count)
print('shipping_null_count',shipping_null_count)
print('item_description_null_count',item_description_null_count)

print("Handling missing values...")
def handle_missing(dataset):
    dataset.category_name.fillna(value="missing/missing/missing", inplace=True)
    dataset.brand_name.fillna(value="missing/missing/missing", inplace=True)
    dataset.item_description.fillna(value="missing/missing/missing", inplace=True)
    return (dataset)

train_df = handle_missing(train_df)
test_df = handle_missing(test_df)
print(train_df.category_name[123:160])


print('count category',train_df['category_name'].value_counts())

#将Category分为3列
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")


train_df['general_cat'], train_df['subcat_1'], train_df['subcat_2'] = zip(*train_df['category_name'].apply(lambda x: split_cat(x)))
test_df['general_cat'], test_df['subcat_1'], test_df['subcat_2'] = zip(*test_df['category_name'].apply(lambda x: split_cat(x)))
print(train_df[130:160])

print("There are %d unique first sub-categories." % train_df['subcat_1'].nunique())
print("There are %d unique second sub-categories." % train_df['subcat_2'].nunique())

print(train_df.columns)
print("Handling categorical variables...")
le = LabelEncoder()

le.fit(np.hstack([train_df.category_name, test_df.category_name]))
train_df.category_name = le.transform(train_df.category_name)
test_df.category_name = le.transform(test_df.category_name)

le.fit(np.hstack([train_df.brand_name, test_df.brand_name]))
train_df.brand_name = le.transform(train_df.brand_name)
test_df.brand_name = le.transform(test_df.brand_name)

le.fit(np.hstack([train_df.name, test_df.name]))
train_df.name = le.transform(train_df.name)
test_df.name = le.transform(test_df.name)

le.fit(np.hstack([train_df.item_description, test_df.item_description]))
train_df.item_description = le.transform(train_df.item_description)
test_df.item_description = le.transform(test_df.item_description)

le.fit(np.hstack([train_df.general_cat, test_df.general_cat]))
train_df.general_cat = le.transform(train_df.general_cat)
test_df.general_cat = le.transform(test_df.general_cat)

le.fit(np.hstack([train_df.subcat_1, test_df.subcat_1]))
train_df.subcat_1 = le.transform(train_df.subcat_1)
test_df.subcat_1 = le.transform(test_df.subcat_1)

le.fit(np.hstack([train_df.subcat_2, test_df.subcat_2]))
train_df.subcat_2 = le.transform(train_df.subcat_2)
test_df.subcat_2 = le.transform(test_df.subcat_2)
del le

dtrain, dvalid = train_test_split(train_df, random_state=123, train_size=0.99)
print(dtrain.shape)
print(dvalid.shape)

#from wordcloud import WordCloud
#from sklearn.feature_extraction.text import TfidfVectorizer
#import string

#cloud = WordCloud(width=1140,height=1080).generate(" ".join(train_df['item_description'].astype(str)))
#plt.figure(figsize=(10,8))
#plt.imshow(cloud)

train_df.drop(['name','item_description'],axis=1,inplace=True)
test_df.drop(['name','item_description'],axis=1,inplace=True)





from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def rmsle1(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5

yy = np.log(y+1)

lasso.fit(train_df, yy)
lasso_train_pred = lasso.predict(test_df)

val_preds = np.exp(lasso_train_pred)+1

submission = pd.DataFrame()
submission['test_id'] = test_id
submission['price'] = val_preds
submission.to_csv('./mysubmission.csv',index=False)

#submission = test_id
#submission["price"] = val_preds
#submission.to_csv("./myNNsubmission.csv", index=False)