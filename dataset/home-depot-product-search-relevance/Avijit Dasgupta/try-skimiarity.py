import time
start_time = time.time()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn import pipeline, grid_search
from sklearn.metrics import mean_squared_error, make_scorer
#from nltk.metrics import edit_distance
from nltk.stem.porter import *
from nltk.corpus import stopwords
import re
import random
import re, math
from collections import Counter



cachedStopWords = stopwords.words("english")
stemmer = PorterStemmer()

WORD = re.compile(r'\w+')


def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

#import enchant

def calculate_similarity(str1,str2):
    vector1 = text_to_vector(str1)
    vector2 = text_to_vector(str2)
    return get_cosine(vector1, vector2)


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
df_pro_desc = pd.read_csv('../input/product_descriptions.csv',encoding="ISO-8859-1")
num_train = df_train.shape[0]

def remove_stopwords(text):
   # text = 'hello bye the the hi'
    return ' '.join([word for word in text.split() if word not in cachedStopWords])
    
def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RSME  = make_scorer(fmean_squared_error, greater_is_better=False)

def str_stem(str1):
    str1 = str1.lower()
    str1 = str1.replace(" in.","in.")
    str1 = str1.replace(" inch","in.")
    str1 = str1.replace("inch","in.")
    str1 = str1.replace(" in ","in. ")
    str1=str1.replace(" ft.","ft.");
    str1=str1.replace(" feet","ft.");
    str1=str1.replace("feet","ft.");
    str1=str1.replace(" ft","ft.");
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
#df_all['search_term'] = df_all['search_term'].map(lambda x:remove_stopwords(x))
#df_all['product_title'] = df_all['product_title'].map(lambda x:remove_stopwords(x))
#df_all['product_description'] = df_all['product_description'].map(lambda x:remove_stopwords(x))

df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
#df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
#df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title'] +"\t"+df_all['product_description']
df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))
df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

df_all['similarity_in_description']=df_all['product_info'].map(lambda x:calculate_similarity(x.split('\t')[0],x.split('\t')[2]))
df_all['similarity_in_title']=df_all['product_info'].map(lambda x:calculate_similarity(x.split('\t')[0],x.split('\t')[1]))

#df_all.to_csv("df_all2.csv")  #no need to keep reprocessing for further grid searches
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
param_grid = {'rfr__n_estimators' : list(range(22,26,1)), 'rfr__max_depth': list(range(6,9,1))}
model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1, cv = 2, verbose = 20, scoring=RSME)
model.fit(X_train, y_train)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

y_pred = model.predict(X_test)
print(len(y_pred))
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission3.csv',index=False)
print("--- Training & Testing: %s minutes ---" % ((time.time() - start_time)/60))
                