import time
start_time = time.time()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import pipeline, grid_search
from sklearn.metrics import mean_squared_error, make_scorer
#from nltk.metrics import edit_distance
from nltk.stem.porter import *
stemmer = PorterStemmer()
import re
#import enchant
import random

random.seed(2016)

df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('../input/product_descriptions.csv')
df_attr = pd.read_csv('../input/attributes.csv')
df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
#Make brand a feature using dictinary, dummies, or vectorizer
num_train = df_train.shape[0]

def str_stem(s): 
    if isinstance(s, str):
        s = s.lower()
        s = s.replace("'","in.") # character
        s = s.replace("inches","in.") # whole word
        s = s.replace("inch","in.") # whole word
        s = s.replace(" in ","in. ") # no period
        s = s.replace(" in.","in.") # prefix space

        s = s.replace("''","ft.") # character
        s = s.replace(" feet ","ft. ") # whole word
        s = s.replace("feet","ft.") # whole word
        s = s.replace("foot","ft.") # whole word
        s = s.replace(" ft ","ft. ") # no period
        s = s.replace(" ft.","ft.") # prefix space
    
        s = s.replace(" pounds ","lb. ") # character
        s = s.replace(" pound ","lb. ") # whole word
        s = s.replace("pound","lb.") # whole word
        s = s.replace(" lb ","lb. ") # no period
        s = s.replace(" lb.","lb.") 
        s = s.replace(" lbs ","lb. ") 
        s = s.replace("lbs.","lb.") 
    
        s = s.replace("*"," xby ")
        s = s.replace(" by"," xby")
        s = s.replace("x0"," xby 0")
        s = s.replace("x1"," xby 1")
        s = s.replace("x2"," xby 2")
        s = s.replace("x3"," xby 3")
        s = s.replace("x4"," xby 4")
        s = s.replace("x5"," xby 5")
        s = s.replace("x6"," xby 6")
        s = s.replace("x7"," xby 7")
        s = s.replace("x8"," xby 8")
        s = s.replace("x9"," xby 9")
    
        s = s.replace(" sq ft","sq.ft. ") 
        s = s.replace("sq ft","sq.ft. ")
        s = s.replace("sqft","sq.ft. ")
        s = s.replace(" sqft ","sq.ft. ") 
        s = s.replace("sq. ft","sq.ft. ") 
        s = s.replace("sq ft.","sq.ft. ") 
        s = s.replace("sq feet","sq.ft. ") 
        s = s.replace("square feet","sq.ft. ") 
    
        s = s.replace(" gallons ","gal. ") # character
        s = s.replace(" gallon ","gal. ") # whole word
        s = s.replace("gallons","gal.") # character
        s = s.replace("gallon","gal.") # whole word
        s = s.replace(" gal ","gal. ") # character
        s = s.replace(" gal","gal") # whole word

        s = s.replace(" ounces","oz.")
        s = s.replace(" ounce","oz.")
        s = s.replace("ounce","oz.")
        s = s.replace(" oz ","oz. ")

        s = s.replace(" centimeters","cm.")    
        s = s.replace(" cm.","cm.")
        s = s.replace(" cm ","cm. ")
        
        s = s.replace(" milimeters","mm.")
        s = s.replace(" mm.","mm.")
        s = s.replace(" mm ","mm. ")
    
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        #volts, watts, amps
        return s.lower()
    else:
        return "null"

def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt

def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stem(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))

df_all['brand'] = df_all['brand'].map(lambda x:str_stem(x))
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_brand'] = df_all['brand'].map(lambda x:len(x.split())).astype(np.int64)
df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title'] +"\t"+df_all['product_description']
df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))
df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']
df_all['attr'] = df_all['search_term']+"\t"+df_all['brand']
df_all['word_in_brand'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['ratio_brand'] = df_all['word_in_brand']/df_all['len_of_brand']
df_all.to_csv('submission.csv',index=False)
print("--- Training & Testing: %s minutes ---" % ((time.time() - start_time)/60))