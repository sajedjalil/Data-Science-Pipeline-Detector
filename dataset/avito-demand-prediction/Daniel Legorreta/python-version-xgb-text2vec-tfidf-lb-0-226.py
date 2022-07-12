# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#Python Version of Konrad Banachewicz 
#https://www.kaggle.com/konradb/xgb-text2vec-tfidf-clone-lb-0-226/versions

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
import xgboost as xgb
import gc


#Load Data
print("Load Data")

stopWords = stopwords.words('russian')
tr =pd.read_csv("../input/train.csv") 
te =pd.read_csv("../input/test.csv")


#Preprocessing
print("Preprocessing")

tri=tr.shape[0]
y = tr.deal_probability.copy()


List_Var=['item_id', 'user_id', 'city', 'param_1', 'param_2', 'param_3','title', 'description', 'activation_date', 'image']

tr_te=tr[tr.columns.difference(["deal_probability"])].append(te)\
     .assign( category_name=lambda x: pd.Categorical(x['category_name']).codes,
              parent_category_name=lambda x:pd.Categorical(x['parent_category_name']).codes,
              region=lambda x:pd.Categorical(x['region']).codes,
              user_type=lambda x:pd.Categorical(x['user_type']).codes,
              price=lambda x: np.log1p(x['price'].fillna(0)),
              txt=lambda x:(x['city'].astype(str)+' '+x['param_1'].astype(str)+' '+x['param_2'].astype(str)+' '+
                           x['param_3'].astype(str)+' '+x['title'].astype(str)+' '+x['description'].astype(str)),
             mon=lambda x: pd.to_datetime(x['activation_date']).dt.month,
             mday=lambda x: pd.to_datetime(x['activation_date']).dt.day,
             week=lambda x: pd.to_datetime(x['activation_date']).dt.week,
             wday=lambda x:pd.to_datetime(x['activation_date']).dt.dayofweek,
            image_top_1=lambda x:x['image_top_1'].fillna(-1))\
             .drop(labels=List_Var,axis=1)
tr_te.price.replace(to_replace=[np.inf, -np.inf,np.nan], value=-1,inplace=True)


del tr,te
gc.collect()

tr_te.loc[:,'txt']=tr_te.txt.apply(lambda x:x.lower().replace("[^[:alpha:]]"," ").replace("\\s+", " "))

print("Processing Text")
def tokenizeL(text):
    return [ w for w in str(text).split()]
    
vec=TfidfVectorizer(ngram_range=(1,3),stop_words=stopWords,min_df=5,max_df=0.3,sublinear_tf=True,norm='l2',max_features=5000)
m_tfidf=vec.fit_transform(tr_te.txt)

tr_te.drop(labels=['txt'],inplace=True,axis=1)

print("General Data")
data  = hstack((tr_te.values,m_tfidf)).tocsr()

del tr_te,m_tfidf
gc.collect()

dtest=xgb.DMatrix(data=data[tri:],missing=-1)
X=data[:tri]
del data
gc.collect()


dtrain =xgb.DMatrix(data = X[:(tri-150342)], label = y[:(tri-150342)],missing=-1)
dval =xgb.DMatrix(data = X[(tri-150342):], label = y[(tri-150342):],missing=-1)
watchlist = [(dval, 'eval')]
del(X, y, tri)
gc.collect()

Dparam = {'objective' : "reg:logistic",
          'booster' : "gbtree",
          'eval_metric' : "rmse",
          'nthread' : 4,
          'eta':0.07,
          'max_depth':18,
          'min_child_weight': 2,
          'gamma' :0,
          'subsample':0.7,
          'colsample_bytree':0.7,
          'aplha':0,
          'lambda':0,
          'nrounds' : 1700}        

print("Training Model")
m_xgb=xgb.train(params=Dparam,dtrain=dtrain,num_boost_round=Dparam['nrounds'],early_stopping_rounds=50,evals=watchlist)

print("Output Model")
(pd.read_csv("../input/sample_submission.csv")
   .assign(deal_probability =m_xgb.predict(dtest))
   .to_csv("baseline_xgb_3.csv", index=False))