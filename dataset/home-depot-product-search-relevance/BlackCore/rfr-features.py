import time
start_time = time.time()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#from sklearn import pipeline, model_selection
from sklearn import pipeline, grid_search
#from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
#from nltk.metrics import edit_distance
from nltk.stem.porter import *
stemmer = PorterStemmer()
#from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
#stemmer = SnowballStemmer('english')
import re
#import enchant
import random
random.seed(8)

df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('../input/product_descriptions.csv')
df_attr = pd.read_csv('../input/attributes.csv')
df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
num_train = df_train.shape[0]
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time)/60),2))

def str_stem(s): 
    if isinstance(s, str):
        s = s.lower()
        s = s.replace(".",". ")
        for s1 in range(0,10):
            s = s.replace(". " + str(s1),"." + str(s1))
        #s = (" ").join([stemmer.stem(z) for z in s.lower().split(" ")])
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        return s.lower()
    else:
        return "null"

def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9]"," ", str2)
    str2 = set(str2.split())
    words = str1.lower().split(" ")
    s = []
    for word in words:
        s.append(segmentit(word,str2,True))
    return (" ".join(s))

def segmentit(s, txt_arr, t):
    st = s
    r = []
    n = s.lstrip('0123456789./')
    if len(n)<len(s):
        r.append(s[0:len(s)-len(n)])
        s=s[len(s)-len(n):len(s)]
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[0:-j]:
                r.append(s[0:-j])
                s=s[len(s)-j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i==len(st):
            r.append(st[i:])
    return (" ".join(r))

stop_w = ['for', 'xbi', 'and', 'x','in', 'th','a','on','s','t','h','w','sku','with','what','from','that','in.','ft.','sq.ft.','gal.','oz.','cm.','mm.','m.','deg.','volt.','watt.','amp.'] #'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':0}
alphab ="abcdefghijklmonpqrstuvwxyz"
numspac = "/."
def str_stem2(s):
    s = s.lower()
    s = s.replace(","," ")
    s = s.replace("$"," ")
    s = s.replace("?"," ")
    s = s.replace("-"," ")
    s = s.replace("//","/")
    s = s.replace("..",".")
    s = s.replace(" / "," ")
    s = s.replace(" \\ "," ")
    
    for s1 in alphab:
        for s2 in numspac:
            s = s.replace(s2+s1,s1+" ")
            s = s.replace(s1+s2,s1+" ")

    for s1 in range(0,10):
        for s2 in alphab:
            s = s.replace(str(s1)+str(s2),str(s1)+" "+str(s2))
            s = s.replace(str(s2)+str(s1),str(s2)+" "+str(s1))

    s = s.replace(" x "," xbi ")
    s = s.replace("*"," xbi ")
    s = s.replace(" by "," xbi ")
    s = s.replace("  "," ")

    s = s.replace("'"," in. ")
    s = s.replace(" in "," in. ")  
    s = s.replace(" inches"," in. ") 
    s = s.replace(" inch"," in. ")

    s = s.replace("''"," ft. ") 
    s = s.replace(" ft "," ft. ") 
    s = s.replace("feet"," ft. ") 
    s = s.replace("foot"," ft. ")

    s = s.replace(" lb"," lb. ")
    s = s.replace(" lbs"," lb. ") 
    s = s.replace("pounds"," lb. ")
    s = s.replace("pound"," lb. ") 

    s = s.replace("sq ft"," sq. ft. ") 
    s = s.replace("sqft"," sq. ft. ")
    s = s.replace("sq. ft"," sq. ft. ") 
    s = s.replace("sq ft."," sq. ft. ") 
    s = s.replace("sq feet"," sq. ft. ") 
    s = s.replace("square feet"," sq. ft. ") 

    s = s.replace("gallons"," gal. ") 
    s = s.replace("gallon"," gal. ") 
    s = s.replace(" gal "," gal. ") 

    s = s.replace(" ounces "," oz. ")
    s = s.replace(" ounce "," oz. ")

    s = s.replace("centimeters"," cm. ")    
    s = s.replace("milimeters"," mm. ")
    s = s.replace("meters"," m. ")

    s = s.replace("°"," deg. ")
    s = s.replace(" degrees"," deg. ")
    s = s.replace(" degree"," deg. ")

    s = s.replace("volts"," volt. ")
    s = s.replace("volt"," volt. ")
    s = s.replace(" v "," volt. ")

    s = s.replace(" watts"," watt. ")
    s = s.replace(" watt"," watt. ")

    s = s.replace(" ampere"," amp. ")
    s = s.replace(" amps"," amp. ")
    s = s.replace(" amp "," amp. ")

    s = str_stem(s)
    s = s.replace("toliet","toilet")
    s = s.replace("airconditioner","air conditioner")
    s = s.replace("vinal","vinyl")
    s = s.replace("vynal","vinyl")
    s = s.replace("skill","skil")
    s = s.replace("snowbl","snow bl")
    s = s.replace("plexigla","plexi gla")
    s = s.replace("rustoleum","rust-oleum")
    s = s.replace("whirpool","whirlpool")
    s = s.replace("whirlpoolga", "whirlpool ga")
    s = s.replace("whirlpoolstainless","whirlpool stainless")

    s = s.replace("  "," ")
    s = s.replace("..",".")
    s = (" ").join([z for z in s.split(" ") if z not in stop_w])
    s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])

    if len(s)<1:
        s = "null"
    return s.lower()

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

class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['id','relevance','search_term','product_title','product_description','product_info','attr','brand']
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        return hd_searches

class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)

#comment out the lines below use df_all.csv for further grid search testing
#if adding features consider any drops on the 'cust_regression_vals' class
#*** would be nice to have a file reuse option or script chaining option on Kaggle Scripts ***
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stem2(x))
print("--- Search Term Stem: %s minutes ---" % round(((time.time() - start_time)/60),2))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))
df_all['brand'] = df_all['brand'].map(lambda x:str_stem(x))
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_brand'] = df_all['brand'].map(lambda x:len(x.split())).astype(np.int64)
df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title'] +"\t"+df_all['product_description']
print("--- Other Feature Stem: %s minutes ---" % round(((time.time() - start_time)/60),2))
df_all['search_term'] = df_all['product_info'].map(lambda x:seg_words(x.split('\t')[0],x.split('\t')[2]))
print("--- Search Term Segment: %s minutes ---" % round(((time.time() - start_time)/60),2))
df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))
print("--- Query In: %s minutes ---" % round(((time.time() - start_time)/60),2))
df_all['query_last_word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[1]))
df_all['query_last_word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[2]))
print("--- Query Last Word In: %s minutes ---" % round(((time.time() - start_time)/60),2))
df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']
df_all['attr'] = df_all['search_term']+"\t"+df_all['brand']
df_all['word_in_brand'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['ratio_brand'] = df_all['word_in_brand']/df_all['len_of_brand']
df_brand = pd.unique(df_all.brand.ravel())
d={}
i = 1
for s in df_brand:
    d[s]=i
    i+=1
df_all['brand_feature'] = df_all['brand'].map(lambda x:d[x])
df_all['search_term_feature'] = df_all['search_term'].map(lambda x:len(x))
df_all.to_csv('df_all.csv')
#df_all = pd.read_csv('df_all.csv', encoding="ISO-8859-1", index_col=0)
df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']
y_train = df_train['relevance'].values
X_train =df_train[:]
X_test = df_test[:]
print("--- Features Set: %s minutes ---" % round(((time.time() - start_time)/60),2))

rfr = RandomForestRegressor(n_estimators = 500, n_jobs = -1, random_state = 8, verbose = 0)
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
tsvd = TruncatedSVD(n_components = 25, random_state = 8, algorithm = 'arpack', tol = 0)
clf = pipeline.Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                        ('cst',  cust_regression_vals()),  
                        ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                        ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                        #('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                        ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)]))
                        ],
                    transformer_weights = {
                        'cst': 1.0,
                        'txt1': 1.0,
                        'txt2': 1.0,
                        #'txt3': 0.5,
                        'txt4': 1.0
                        },
                #n_jobs = -1
                )), 
        ('rfr', rfr)])
param_grid = {'rfr__max_features': [25], 'rfr__max_depth': [18]}
model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1, cv = 2, verbose = 0, scoring=RMSE)
model.fit(X_train, y_train)
print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)
print(model.best_score_ + 0.47003199274)

y_pred = model.predict(X_test)
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)
print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time)/60),2))
'''
from sklearn.feature_extraction import text
import nltk
import re
import numpy as np
import pandas as pd

df_outliers = pd.read_csv('df_all.csv', encoding="ISO-8859-1", index_col=0)

def str_common_word(str1, str2):
    str1 = str(str1).lower()
    str2 = str(str2).lower()
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt

#stop_ = list(text.ENGLISH_STOP_WORDS)
stop_ = []
d={}
for i in range(len(df_outliers)):
    s = str(df_outliers['search_term'][i]).lower()
    #s = s.replace("\n"," ")
    #s = re.sub("[^a-z]"," ", s)
    #s = s.replace("  "," ")
    a = set(s.split(" "))
    for b_ in a:
        if b_ not in stop_ and len(b_)>0:
            if b_ not in d:
                d[b_] = [1,str_common_word(b_, df_outliers['product_title'][i]),str_common_word(b_, df_outliers['brand'][i]),str_common_word(b_, df_outliers['product_description'][i])]
            else:
                d[b_][0] += 1
                d[b_][1] += str_common_word(b_, df_outliers['product_title'][i])
                d[b_][2] += str_common_word(b_, df_outliers['brand'][i])
                d[b_][3] += str_common_word(b_, df_outliers['product_description'][i])
ds2 = pd.DataFrame.from_dict(d,orient='index')
ds2.columns = ['count','in title','in brand','in prod']
ds2 = ds2.sort_values(by=['count'], ascending=[False])

f = open("word_review.csv", "w")
f.write("word|count|in title|in brand|in description\n")
for i in range(len(ds2)):
    f.write(ds2.index[i] + "|" + str(ds2["count"][i]) + "|" + str(ds2["in title"][i]) + "|" + str(ds2["in brand"][i]) + "|" + str(ds2["in prod"][i]) + "\n")
f.close()
print("--- Word List Created: %s minutes ---" % round(((time.time() - start_time)/60),2))
'''