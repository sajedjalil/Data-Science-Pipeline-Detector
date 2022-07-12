# Dataframe created as df_all.bin

import time
import numpy as np
import pandas as pd
from nltk.stem.porter import *
stemmer = PorterStemmer()
import re
import random

start_time = time.time()

random.seed(1966)

df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('../input/product_descriptions.csv')
df_attr = pd.read_csv('../input/attributes.csv')
df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})

num_train = df_train.shape[0]

def str_stem(s): 
    if isinstance(s, str):
        #s = s.replace("'","in.") # character
        s = s.replace("inches","in.") # whole word
        s = s.replace("inch","in.") # whole word
        s = s.replace(" in ","in. ") # no period
        s = s.replace(" in.","in.") # prefix space

        #s = s.replace("''","ft.") # character
        s = s.replace(" feet ","ft. ") # whole word
        s = s.replace("feet","ft.") # whole word
        s = s.replace("foot","ft.") # whole word
        s = s.replace(" ft ","ft. ") # no period
        s = s.replace(" ft.","ft.") # prefix space
        
        s = s.replace(" pounds ","lb. ") # character
        s = s.replace(" pound ","lb. ") # whole word
        s = s.replace("pound","lb.") # whole word
        s = s.replace(" lb ","lb. ") # no period
        s = s.replace(" lb.","lb.") # prefix space
        
        #regex for 4x2 split and UoM
        #volts, watts, amps, mm, sq ft, cm, gallons
        
        return " ".join([stemmer.stem(re.sub('[^A-Za-z0-9-./]', ' ', word)) for word in s.lower().split()])
    else:
        return "null"

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

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')

print("merged")

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
df_all['query_title_len_prop']=df_all['len_of_title']/df_all['len_of_query']
df_all['query_desc_len_prop']=df_all['len_of_description']/df_all['len_of_query']
df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']

df_all['attr'] = df_all['search_term']+"\t"+df_all['brand']+"\t"+df_all['product_title']
df_all['word_in_brand'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['ratio_brand'] = df_all['word_in_brand']/df_all['len_of_brand']

print("calculated df_all")

df_all = df_all.drop(['search_term','product_title','product_description','product_info','attr','brand'],axis=1)
df_all.head()
print (df_all[:10])
#df_all.to_csv("df_all2.csv")  #no need to keep reprocessing for further grid searches
print("--- Features Set: %s minutes ---" % ((time.time() - start_time)/60))

df_all.to_pickle("df_all.bin") 

print("Dataframe created!")