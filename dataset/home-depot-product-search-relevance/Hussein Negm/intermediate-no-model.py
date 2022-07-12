import time
start_time = time.time()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import pipeline, grid_search
from sklearn.metrics import mean_squared_error, make_scorer
#from nltk.metrics import edit_distance
from nltk.stem.porter import *
stemmer = PorterStemmer()
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
#import enchant
import random
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
random.seed(313)
"""
def str_stem(str1):
    str1 = str1.lower()
    str1 = str1.replace(" in.","in.")
    str1 = str1.replace(" inch","in.")
    str1 = str1.replace("inch"," in.")
    str1 = str1.replace(" in "," in. ")
    str1 = str1.replace(" ' "," in. ")
    str1 = str1.replace("*","x")
    str1 = str1.replace(",","")
    str1 = str1.replace("height","h")
    str1 = str1.replace("width","w")
    str1 = str1.replace("-volt","v")
    str1 = str1.replace("r-","r")
   
    str1 = (" ").join([stemmer.stem(z) for z in str1.split(" ")])
    return str1
    
"""

def str_stem(str1):
    str1 = str1.lower()
    str1 = str1.replace(" in.","")
    str1 = str1.replace(" inch","")
    str1 = str1.replace("inch","")
    str1 = str1.replace(" in ","")
    str1 = str1.replace("0in","0 in.")
    str1 = str1.replace("1in","1 in.")
    str1 = str1.replace("2in","2 in.")
    str1 = str1.replace("3in","3 in.")
    str1 = str1.replace("4in","4 in.")
    str1 = str1.replace("5in","5 in.")
    str1 = str1.replace("6in","6 in.")
    str1 = str1.replace("7in","7 in.")
    str1 = str1.replace("8in","8 in.")
    str1 = str1.replace("9in","9 in.")
    str1 = str1.replace("'"," in.")
    str1 = str1.replace("*"," x ")
    str1 = str1.replace("x0"," x 0")
    str1 = str1.replace("x1"," x 1")
    str1 = str1.replace("x2"," x 2")
    str1 = str1.replace("x3"," x 3")
    str1 = str1.replace("x4"," x 4")
    str1 = str1.replace("x5"," x 5")
    str1 = str1.replace("x6"," x 6")
    str1 = str1.replace("x7"," x 7")
    str1 = str1.replace("x8"," x 8")
    str1 = str1.replace("x9"," x 9")
    str1 = str1.replace("-"," ")
    str1 = str1.replace(",","")
    str1 = str1.replace("height","h")
    str1 = str1.replace("width","w")
#    str1 = str1.replace("ft.","")
#    str1 = str1.replace("sq.","")
#    str1 = str1.replace("oz.","")
#    str1 = str1.replace("cu.","")
#    str1 = str1.replace("lb.","")
#    str1 = str1.replace("gal.","")
    str1 = str1.replace("by"," x ")
    str1 = str1.replace("-volt","v")
    str1 = str1.replace("r-","r")
   
    str1 = (" ").join([stemmer.stem(z) for z in str1.split(" ")])
    return str1
   

def str_common_word(str1, str2):
    str1, str2 = str1.lower(), str2.lower()
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt

def str_whole_word(str1, str2, i_):
    str1, str2 = str1.lower().strip(), str2.lower().strip()
    cnt = 0
    #if len(str1)>0 and len(str2)>0:
    #    cnt = len(re.findall(str1,str2))
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt

df_train = pd.read_csv('../input/train.csv' , encoding='ISO-8859-1')
df_test = pd.read_csv('../input/test.csv' , encoding='ISO-8859-1' )
df_desc = pd.read_csv("../input/product_descriptions.csv" , encoding='utf-8')




## REMOVE STOP words
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))
df_train['product_title'] = df_train['product_title'].apply(lambda x: " ".join([item for item in x.split() if item.lower() not in stops]))
df_train['search_term'] = df_train['search_term'].apply(lambda x: " ".join([item for item in x.split() if item.lower() not in stops]))

df_test['product_title'] = df_test['product_title'].apply(lambda x: " ".join([item for item in x.split() if item.lower() not in stops]))
df_test['search_term'] = df_test['search_term'].apply(lambda x: " ".join([item for item in x.split() if item.lower() not in stops]))

df_desc['product_description'] = df_desc['product_description'].apply(lambda x: " ".join([item for item in x.split() if item.lower() not in stops]))


#attr = attr[attr['value'] != 'No']
#df_brands = attr[attr['name']== 'MFG Brand Name']
#df_brands= df_brands.drop('name', axis=1)
#df_brands= df_brands.rename(columns={'value': 'brand_name'})
#df_attr = attr.groupby('product_uid').apply(func2)

num_train = df_train.shape[0]
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_desc, how='left', on='product_uid')
#df_all = pd.merge(df_all, df_brands, how='left', on='product_uid')
#df_all = pd.merge(df_all, df_attr, how='left', on='product_uid')

#df_all['brand_name'] = df_all['brand_name'].str.decode('utf-8')
#df_all['names'] = df_all['names'].str.decode('utf-8')
#df_all['values'] = df_all['values'].str.decode('utf-8')


df_all[['search_term','product_title','product_description']] = df_all[['search_term','product_title','product_description']].applymap(lambda x:str_stem(x))
#df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
#df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title'] +"\t"+df_all['product_description']# +"\t"+df_all['brand_name'] +"\t"+df_all['names'] +"\t"+df_all['values'] 
#df_all['product_info'] = df_all['product_info'].astype(str)
df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))
df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']


df_all.to_csv("df_allStop.csv", encoding="utf-8", index=False)
