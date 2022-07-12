# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import pandas as pd  #pandas for using dataframe and reading csv 
import numpy as np   #numpy for vector operations and basic maths 
import urllib        #for url stuff
import re            #for processing regular expressions
import datetime      #for datetime operations
import calendar      #for calendar for datetime operations
import time          #to get the system time
import scipy         #for other dependancies
from sklearn.cluster import KMeans # for doing K-means clustering
from haversine import haversine # for calculating haversine distance
import math          #for basic maths operations
import seaborn as sns #for making plots
import matplotlib.pyplot as plt # for plotting
import os                # for os commands
import nltk
from nltk.corpus import stopwords
import string
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes
import time
from nltk.stem.porter import *
import multiprocessing as mp



train = pd.read_csv("../input/train.tsv",sep = '\t') # (1482535, 8)
test = pd.read_csv("../input/test.tsv",sep = "\t") #  (693359, 7)

# Deal with missing value
def deal_missing(df):
    df['category_name'] = df['category_name'].fillna('undefined').astype(str)
    df['brand_name'] = df['brand_name'].fillna('undefined').astype(str)
    df['shipping'] = df['shipping'].astype(str)
    df['item_condition_id'] = df['item_condition_id'].astype(str)
    df['item_description'] = df['item_description'].fillna('undefined')
    df.item_description.replace("No description yet","undefined",inplace = True)
    return df
train = deal_missing(train)
test = deal_missing(test)

# Deal with categories
def cat_split(row):
    try:
        text = row
        txt1, txt2, txt3 = text.split('/')
        return txt1, txt2, txt3
    except:
        return "undefined", "undefined", "undefined"
train["C1"],train["C2"],train["C3"] = zip(*train.category_name.map(cat_split))
test["C1"],test["C2"],test["C3"] = zip(*test.category_name.map(cat_split))

# Transform price to log(price+1)
train['log_price'] = np.log(train.price+1)

# Description length
train['des_len']= train.item_description.map(len)
train['des_word_len']=train.item_description.apply(lambda x :len(x.split()))
test['des_len']= test.item_description.map(len)
test['des_word_len']=test  .item_description.apply(lambda x :len(x.split()))

# if rm in description
def rm(row):
    if "rm" in row:
        return 1
    return 0
train["if_rm"] = train.item_description.map(rm)
test["if_rm"] =  test.item_description.map(rm)

# Brand or not
def dummy(row):
    if row=="undefined":
        return 0
    return 1
train["if_brand"] = train.brand_name.map(dummy)
train["if_category"] = train.category_name.map(dummy)
train["if_description"] = train.item_description.map(dummy)

test["if_brand"] = test.brand_name.map(dummy)
test["if_category"] = test.category_name.map(dummy)
test["if_description"] = test.item_description.map(dummy)

# Label brand, category 
key_cat1 = train.C1.unique().tolist() + test.C1.unique().tolist()
key_cat1 = list(set(key_cat1))
value_cat1 = list(range(key_cat1.__len__()))
dict_cat1 = dict(zip(key_cat1, value_cat1))

key_cat2 = train.C2.unique().tolist() + test.C1.unique().tolist()
key_cat2 = list(set(key_cat2))
value_cat2 = list(range(key_cat2.__len__()))
dict_cat2 = dict(zip(key_cat2, value_cat2))

key_cat3 = train.C3.unique().tolist() + test.C3.unique().tolist()
key_cat3 = list(set(key_cat3))
value_cat3 = list(range(key_cat3.__len__()))
dict_cat3 = dict(zip(key_cat3, value_cat3))

key_brand = train.brand_name.unique().tolist()+test.brand_name.unique().tolist()
key_brand = list(set(key_brand))
value_brand = list(range(len(key_brand)))
dict_brand = dict(zip(key_brand, value_brand))

def label(row,dic):
    return dic.get(row)

train['C1_label']=[label(a, dict_cat1) for a in train.C1]
train['C2_label']=[label(a, dict_cat2) for a in train.C2]
train['C3_label']=[label(a, dict_cat3) for a in train.C3]

test['C1_label']=[label(a, dict_cat1) for a in test.C1]
test['C2_label']=[label(a, dict_cat2) for a in test.C2]
test['C3_label']=[label(a, dict_cat3) for a in test.C3]

train['bran_label']=[label(a, dict_brand) for a in train.brand_name]
test['bran_label']=[label(a, dict_brand) for a in test.brand_name]   

# BoW Model for name and description
# Tokenization
start = time.time()
translate_table = dict((ord(char), None) for char in string.punctuation)
def clean(text):
    text = [w for w in text if re.search("\w", w) and len(w)>2 and len(w)<15 and not re.search("\\.",w) and w.isdigit()==0]
    text = [w for w in text if not re.search("[0-9]", w)]
    return text

def get_tokens(text):
    text = text.translate(translate_table)
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = clean(text)
    text = ' '.join(text)
    #remove the punctuation using the character deletion step of translate   
    return text

train["token_des"]= train.item_description.map(get_tokens)
test["token_des"] = test.item_description.map(get_tokens)

# SVD - TF-IDF for description

tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
full_tfidf = tfidf_vec.fit_transform(train.token_des.values.tolist() + test.token_des.values.tolist())
train_tfidf = tfidf_vec.transform(train['token_des'].values.tolist())
test_tfidf = tfidf_vec.transform(test['token_des'].values.tolist())

n_comp = 40
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
train_svd = pd.DataFrame(svd_obj.fit_transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.fit_transform(test_tfidf))
    
train_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]
train = pd.concat([train, train_svd], axis=1)
test = pd.concat([test, test_svd], axis=1)

# SVD - TF-IDF for name
train["token_name"] = train.name.map(get_tokens)
test["token_name"] = test.name.map(get_tokens)

tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
full_name_tfidf = tfidf_vec.fit_transform(train.token_name.values.tolist() + test.token_name.values.tolist())
train_name_tfidf = tfidf_vec.transform(train['token_name'].values.tolist())
test_name_tfidf = tfidf_vec.transform(test['token_name'].values.tolist())

n_comp = 40
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
train_svd = pd.DataFrame(svd_obj.fit_transform(train_name_tfidf))
test_svd = pd.DataFrame(svd_obj.fit_transform(test_name_tfidf))

train_svd.columns = ['svd_name_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_name_'+str(i) for i in range(n_comp)]
train = pd.concat([train, train_svd], axis=1)
test = pd.concat([test, test_svd], axis=1)

# Convert int to str, preparing for later analysis 
 
train_y = train.log_price
train_x = train.drop(["name","category_name","brand_name","item_description",\
                      "C1","C2","C3","log_price","token_des","token_name","train_id",\
                      "price"],axis = 1)

train_x[["item_condition_id","shipping","if_rm","if_brand","if_category","if_description",\
        "C1_label","C2_label","C3_label","bran_label"]]=train_x[["item_condition_id","shipping","if_rm","if_brand","if_category","if_description",\
        "C1_label","C2_label","C3_label","bran_label"]].astype(str)

test_x = test.drop(["name","category_name","brand_name","item_description",\
                      "C1","C2","C3","token_des","token_name",\
                      "test_id"],axis = 1)

test_x[["item_condition_id","shipping","if_rm","if_brand","if_category","if_description",\
        "C1_label","C2_label","C3_label","bran_label"]]=test_x[["item_condition_id","shipping","if_rm","if_brand","if_category","if_description",\
        "C1_label","C2_label","C3_label","bran_label"]].astype(str)
features_train = train_x[["item_condition_id","shipping","des_len","des_word_len",\
                        "if_rm","if_brand","if_description","C1_label","C2_label",\
                        "C3_label","bran_label"]]
features_test = test_x[["item_condition_id","shipping","des_len","des_word_len",\
                        "if_rm","if_brand","if_description","C1_label","C2_label",\
                        "C3_label","bran_label"]]
tfidf_train = train_x.drop(['item_condition_id',
 'shipping',
 'des_len',
 'des_word_len',
 'if_rm',
 'if_brand',
 'if_category',
 'if_description',
 'C1_label',
 'C2_label',
 'C3_label','bran_label'], axis = 1)

tfidf_test = test_x.drop(['item_condition_id',
 'shipping',
 'des_len',
 'des_word_len',
 'if_rm',
 'if_brand',
 'if_category',
 'if_description',
 'C1_label',
 'C2_label',
 'C3_label','bran_label'],axis = 1)
# Model building

# GBRegressor
from sklearn.ensemble import GradientBoostingRegressor
model_GB = GradientBoostingRegressor(n_estimators = 100)
trained_GB = model_GB.fit(features_train, train_y)
pre_GB = model_GB.predict(features_test)
pre_GB = np.exp(pre_GB)-1
pre_GB = pd.Series(pre_GB).map(lambda x: max(x,0))
otpt = pd.concat([pd.Series(range(len(pre_GB))),pd.DataFrame(pre_GB)], axis = 1)
otpt.columns = ["test_id","price"]
otpt.to_csv("otpt.csv", index = False)