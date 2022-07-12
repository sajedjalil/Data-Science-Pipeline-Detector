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

random.seed(23)

df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")
#df_attr = pd.read_csv('../input/attributes.csv')
#df_attr = df_attr.dropna()
#df_attr = df_attr.reset_index(drop=True)
#d = {}
#for i in range(len(df_attr)):
#    #print(df_attr[i:i+1])
#    if str(int(df_attr.product_uid[i])) in d:
#        d[str(int(df_attr.product_uid[i]))][1] += " " + str(df_attr['value'][i]).replace('\t'," ")
#    else:
#        d[str(int(df_attr.product_uid[i]))] = [int(df_attr.product_uid[i]),str(df_attr['value'][i])]
#df_attr = pd.DataFrame.from_dict(d,orient='index')
#df_attr.columns = ['product_uid','value']
#print(df_attr.head())
df_pro_desc = pd.read_csv('../input/product_descriptions.csv')
num_train = df_train.shape[0]

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RSME  = make_scorer(fmean_squared_error, greater_is_better=False)

def str_stem(str1):
	str1 = str1.lower()
	str1 = str1.replace(" in.","in.")
	str1 = str1.replace(" inch","in.")
	str1 = str1.replace(" inches","in.")
	
	str1 = str1.replace(" lb.","lb.")
	str1 = str1.replace(" lb","lb.")
	str1 = str1.replace(" pound","lb.")
	
	str1 = str1.replace(" oz.","oz.")
	str1 = str1.replace(" oz","oz.")
	str1 = str1.replace("ounce","oz.")
	
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

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stem(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title'] +"\t"+df_all['product_description']
df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))
df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']

df_all.to_csv("df_all2.csv")  #no need to keep reprocessing for further grid searches
df_all = df_all.drop(['search_term','product_title','product_description','product_info'],axis=1)
df_all.head()
df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']
y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values
print("--- Features Set: %s minutes ---" % ((time.time() - start_time)/60))
rfr = RandomForestRegressor()
clf = pipeline.Pipeline([('rfr', rfr)])
param_grid = {'rfr__n_estimators' : [109],#list(range(109,110,1)), 
              'rfr__max_depth': [9], #list(range(7,8,1))
              #'rfr__max_features' : [.6,.8],
            }
model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1, cv = 10, verbose = 150, scoring=RSME)
model.fit(X_train, y_train)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

y_pred = model.predict(X_test)
print(len(y_pred))
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission2.csv',index=False)
print("--- Training & Testing: %s minutes ---" % ((time.time() - start_time)/60))