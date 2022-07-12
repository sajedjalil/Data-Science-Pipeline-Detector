# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-notebook-avito v23

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb
from nltk.corpus import stopwords
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

#color = sns.color_palette()
#%matplotlib inline

#import plotly.offline as py
#py.init_notebook_mode(connected=True)
#import plotly.graph_objs as go
#import plotly.tools as tls
from tqdm import tqdm

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
# Any results you write to the current directory are saved as output.

stopWords = stopwords.words('russian')

train_df = pd.read_csv("../input/train.csv", parse_dates=["activation_date"])
test_df = pd.read_csv("../input/test.csv", parse_dates=["activation_date"])
print("Train file rows and columns are : ", train_df.shape)
print("Test file rows and columns are : ", test_df.shape)

# Target and ID variables #
train_y = train_df["deal_probability"].values
test_id = test_df["item_id"].values

# New variable on weekday #
train_df["activation_weekday"] = train_df["activation_date"].dt.weekday
test_df["activation_weekday"] = test_df["activation_date"].dt.weekday

####
data = pd.concat([train_df, test_df], axis=0)

#https://www.kaggle.com/classtag/lightgbm-with-mean-encode-feature-0-233
agg_cols = ['region', 'city', 'parent_category_name', 'category_name', 'image_top_1', 'user_type','item_seq_number','activation_weekday']
for c in tqdm(agg_cols):
    gp = train_df.groupby(c)['deal_probability']
    mean = gp.mean()
    std  = gp.std()
    data[c + '_deal_probability_avg'] = data[c].map(mean)
    data[c + '_deal_probability_std'] = data[c].map(std)

for c in tqdm(agg_cols):
    gp = train_df.groupby(c)['price']
    mean = gp.mean()
    data[c + '_price_avg'] = data[c].map(mean)
#####

tfidf = TfidfVectorizer(max_features=50000, stop_words = stopWords)
tfidf_title = TfidfVectorizer(max_features=50000, stop_words = stopWords)

train_df['description'] = train_df['description'].fillna(' ')
test_df['description'] = test_df['description'].fillna(' ')
train_df['title'] = train_df['title'].fillna(' ')
test_df['title'] = test_df['title'].fillna(' ')
tfidf.fit(pd.concat([train_df['description'], test_df['description']]))
tfidf_title.fit(pd.concat([train_df['title'], test_df['title']]))


train_des_tfidf = tfidf.transform(train_df['description'])
test_des_tfidf = tfidf.transform(test_df['description'])

train_title_tfidf = tfidf.transform(train_df['title'])
test_title_tfidf = tfidf.transform(test_df['title'])

n_comp = 3
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(tfidf.transform(pd.concat([train_df['description'], test_df['description']])))

svd_title = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_title.fit(tfidf.transform(pd.concat([train_df['title'], test_df['title']])))

train_svd = pd.DataFrame(svd_obj.transform(train_des_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_des_tfidf))
train_svd.columns = ['svd_des_'+str(i+1) for i in range(n_comp)]
test_svd.columns = ['svd_des_'+str(i+1) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)

train_title_svd = pd.DataFrame(svd_title.transform(train_title_tfidf))
test_titile_svd = pd.DataFrame(svd_title.transform(test_title_tfidf))
train_title_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
test_titile_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
train_df = pd.concat([train_df, train_title_svd], axis=1)
test_df = pd.concat([test_df, test_titile_svd], axis=1)
###
# Label encode the categorical variables #
cat_vars = ["region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3"]
for col in tqdm(cat_vars):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

cols_to_drop = ["item_id", "user_id", "title", "description", "activation_date", "image"]
print(train_df.columns)
train_X = train_df.drop(cols_to_drop + ["deal_probability"], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

print(train_X.head())

def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 40,
        "learning_rate" : 0.09,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=20, evals_result=evals_result)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result
#[3404]	valid_0's rmse: 0.224523

# Splitting the data for model training#
dev_X = train_X.iloc[:-200000,:]
val_X = train_X.iloc[-200000:,:]
dev_y = train_y[:-200000]
val_y = train_y[-200000:]
print(dev_X.shape, val_X.shape, test_X.shape)

# Training the model #
pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

# Making a submission file #
pred_test[pred_test>1] = 1
pred_test[pred_test<0] = 0
sub_df = pd.DataFrame({"item_id":test_id})
sub_df["deal_probability"] = pred_test
sub_df.to_csv("baseline_lgb.csv", index=False)