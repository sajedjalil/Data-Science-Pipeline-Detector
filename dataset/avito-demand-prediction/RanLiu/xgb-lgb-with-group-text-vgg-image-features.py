# Note: This includes many things from various public kernels,  
# but it does not work very well compared to simpler models

#-------------------------#
# 0. Basic Preparations   #
#-------------------------#

# Import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer # For text feature extraction
from sklearn.decomposition import TruncatedSVD 
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb
from nltk.corpus import stopwords # identify stopwords
from tqdm import tqdm
import time
import gc

import os
print(os.listdir("../input"))

# basic setting
pd.options.mode.chained_assignment = None #turn off warning for chained assignment
pd.options.display.max_columns = 999 #increase columns that can be desplayed
stopWords = stopwords.words('russian')

# Read datasets
train_df = pd.read_csv("../input/avito-demand-prediction/train.csv", parse_dates=["activation_date"])
test_df = pd.read_csv("../input/avito-demand-prediction/test.csv", parse_dates=["activation_date"])
print("Train file rows and columns are : ", train_df.shape)
print("Test file rows and columns are : ", test_df.shape)
train_df.head()
test_df.head()

# Target and ID variables #
train_y = train_df["deal_probability"].values
test_id = test_df["item_id"].values

#### Combine train and test data
data = pd.concat([train_df, test_df], axis=0)
length = len(train_df)

del train_df, test_df
gc.collect()

# New time features #
data["activation_weekday"] = data["activation_date"].dt.weekday

print ("Step 0 finished")

#--------------------------------#
# 1. Simple aggregated features  #
#--------------------------------#
# https://www.kaggle.com/classtag/lightgbm-with-mean-encode-feature-0-233
# map() function returns a list of the results after applying the given function to each item of a given iterable 
# Get mean and standard deviation of price by different groups
agg_cols = ['region', 'city', 'parent_category_name', 'category_name', 'image_top_1', 'user_type','item_seq_number','activation_weekday']

#Get mean price by different groups
for c in tqdm(agg_cols):
    gp = data.groupby(c)['price']
    mean = gp.mean()
    std = gp.std()
    data[c + '_price_avg'] = data[c].map(mean)
    data[c + '_price_std'] = data[c].map(std)
    
gc.collect()
print ("Step 1 finished")


#----------------------------#
# 2.Text Feature Extraction  #
#----------------------------#

# TfidfVectorizer convert a collection of raw documents to a matrix of TF-IDF features
# Tf means term-frequency while tf–idf means term-frequency times inverse document-frequency

tfidf_des = TfidfVectorizer(max_features=6500, stop_words = stopWords, max_df=0.4)
tfidf_title = TfidfVectorizer(max_features=6500, stop_words = stopWords, max_df=0.4)

# Fill missing values in description and title 
data['description'] = data['description'].fillna(' ')
data['title'] = data['title'].fillna(' ')

data['description'] = data['description'].str.replace("[^[:alpha:]]", " ")
data['title'] = data['title'].str.replace("[^[:alpha:]]", " ")

data['description'] = data['description'].str.replace("\\s+", " ")
data['title'] = data['title'].str.replace("\\s+", " ")


# fit and transform Russian description and title
tfidf_des.fit(data['description'])
tfidf_title.fit(data['title'])

des_tfidf = tfidf_des.transform(data['description'])
title_tfidf = tfidf_title.transform(data['title'])

gc.collect()

print ("Step 2 finished")

#----------------------------#
# 3.Dimensionality reduction #
#----------------------------#
# find best n_comp
# des: n_components = 
# title: n_component = 
n_comp=30

svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(des_tfidf)

svd_title = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_title.fit(title_tfidf)

data_des_svd = pd.DataFrame(svd_obj.transform(des_tfidf))
data_des_svd.columns = ['svd_des_'+str(i+1) for i in range(n_comp)]
data[data_des_svd.columns] = data_des_svd 

data_title_svd = pd.DataFrame(svd_title.transform(title_tfidf))
data_title_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
data[data_title_svd.columns] = data_title_svd

del des_tfidf, title_tfidf, data_des_svd, data_title_svd
gc.collect()

print ("Step 3 finished")

#---------------------------------#
# 4.Count features for text data  #
#---------------------------------#

from textblob import TextBlob
import string

punctuation = string.punctuation
stop_words = list(set(stopwords.words('russian')))

data['d_length'] = data['description'].apply(lambda x: len(str(x))) 
data['char_count'] = data['description'].apply(len)
data['word_count'] = data['description'].apply(lambda x: len(x.split()))
data['word_density'] = data['char_count'] / (data['word_count']+1)
data['punctuation_count'] = data['description'].apply(lambda x: len("".join(_ for _ in x if _ in punctuation))) 
data['title_word_count'] = data['description'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
data['upper_case_word_count'] = data['description'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
data['stopword_count'] = data['description'].apply(lambda x: len([wrd for wrd in x.split() if wrd.lower() in stop_words]))

data.columns
gc.collect()

print("Step 4 finished")

#---------------------------------#
# 5.Image features                #
#---------------------------------#
# https://www.kaggle.com/wesamelshamy/ad-image-recognition-and-quality-scoring
# https://www.kaggle.com/sohier/getting-started-loading-the-images
# https://www.kaggle.com/bguberfain/naive-lgb-with-text-images
debug = False

from scipy import sparse
import gzip
from pathlib import PurePath


# Create a function to load image features
def load_imfeatures(folder):
    path = PurePath(folder)
    features = sparse.load_npz(str(path / 'features.npz'))
    
    if debug:
        features = features[:100000]
        
    return features
    
  
ftrain = load_imfeatures('../input/vgg16-train-features/')
ftest = load_imfeatures('../input/vgg16-test-features/')


# Merge two datasets
fboth = sparse.vstack([ftrain, ftest])
del ftrain, ftest
gc.collect()
assert fboth.shape[0]==data.shape[0]

# Categorical image feature (max and min VGG16 feature)
data['im_max_feature'] = fboth.argmax(axis=1)  # This will be categorical
data['im_min_feature'] = fboth.argmin(axis=1)  # This will be categorical

data['im_n_features'] = fboth.getnnz(axis=1)
data['im_mean_features'] = fboth.mean(axis=1)
data['im_meansquare_features'] = fboth.power(2).mean(axis=1)

# Reduce image features
tsvd = TruncatedSVD(32)
ftsvd = tsvd.fit_transform(fboth)
del fboth
gc.collect()

# Merge image features into data
df_ftsvd = pd.DataFrame(ftsvd, index=data.index).add_prefix('im_tsvd_')
data = pd.concat([data, df_ftsvd], axis=1)
del df_ftsvd, ftsvd
gc.collect()

print ("Step 5 finished")

#---------------------------------#
# 6.Other work before modeling    #
#---------------------------------#

# Label encode the categorical variables #
cat_vars = ["region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3"]
for col in tqdm(cat_vars):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(data[col].values.astype('str')))
    data[col] = lbl.transform(list(data[col].values.astype('str')))

print(data.columns)

# Drop unuseful columns
cols_to_drop = ["item_id", "user_id", "title", "description", "activation_date", "image"]

# Split train and test data
train = data[:length]
test = data[length:]

train_X = train.drop(cols_to_drop + ["deal_probability"], axis=1)
test_X = test.drop(cols_to_drop+ ["deal_probability"], axis=1)

del train, test
gc.collect

print ("Step 6 finished")

#---------------------------------#
# 7. Light GBM Modeling           #
#---------------------------------#

# Splitting the data for model training#
from sklearn.model_selection import train_test_split
dev_X, val_X, dev_y, val_y = train_test_split(train_X, train_y, test_size = 0.2, random_state = 42)
#dev_X = train_X.iloc[:-200000,:]
#val_X = train_X.iloc[-200000:,:]
#dev_y = train_y[:-200000]
#val_y = train_y[-200000:]

print(dev_X.shape, val_X.shape, test_X.shape)

def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 40,
        "learning_rate" : 0.05,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1,
        "random_state": 42
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                      valid_sets=[lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=50, 
                      evals_result=evals_result)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result

# Training the model #
pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

print("prediction completed")

# feature importance
print("Features importance...")
gain = model.feature_importance('gain')
ft = pd.DataFrame({'feature':model.feature_name(), 
                   'split':model.feature_importance('split'), 
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(ft)


# [4141]	valid_0's rmse: 0.222977
# Added image feature [2791]	valid_0's rmse: 0.223854
# Removed group price feature valid_0's rmse: 0.225456
# Changed validation set: 0.223851

#-----------------#
# 8. XGB Modeling #
#-----------------#
import xgboost as xgb

params = {'objective': 'reg:logistic', 
          'eval_metric': 'rmse', 
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'eta': 0.05,
          'max_leaves': 1400,  
          'max_depth': 17, 
          'subsample': 0.7, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':1,
          'alpha':0,
          'random_state': 42, 
          'silent': True}

tr_data = xgb.DMatrix(dev_X, dev_y)
va_data = xgb.DMatrix(val_X, val_y)

watchlist = [(tr_data, 'train'), (va_data, 'valid')]

model_xgb = xgb.train(params, tr_data, 2000, watchlist, maximize=False, early_stopping_rounds = 30, verbose_eval=10)

#valid-rmse:0.224304
#after adding image features: valid-rmse:0.225647
#after removing mean and variances: 0.227187
#after changing validation set: 0.222999
#Increase text features: 

dtest = xgb.DMatrix(test_X)
xgb_pred_y = model_xgb.predict(dtest, ntree_limit=model_xgb.best_ntree_limit)


#----------------------------#
# 9. Writing submission file #
#----------------------------#

pred_test[pred_test>1] = 1
pred_test[pred_test<0] = 0
xgb_pred_y[xgb_pred_y>1] = 1
xgb_pred_y[xgb_pred_y<0] = 0

sub_lgb = pd.DataFrame({"item_id":test_id})
sub_lgb["deal_probability"] = pred_test
sub_lgb.to_csv("lgb_v4.csv", index=False)

sub_xgb = pd.DataFrame({"item_id":test_id})
sub_xgb["deal_probability"] = xgb_pred_y
sub_xgb.to_csv("xgb_v4.csv", index=False)
