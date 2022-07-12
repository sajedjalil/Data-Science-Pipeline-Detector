# Light GBM for Avito Demand Prediction Challenge
#The base of script is  https://www.kaggle.com/konradb/xgb-text2vec-tfidf-clone-lb-0-226/versions (Konrad Banachewicz)


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
import lightgbm as gbm
import gc
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse

#Load Data
print("Load Data")

stopWords = stopwords.words('russian')
tr =pd.read_csv("../input/train.csv") 
te =pd.read_csv("../input/test.csv")


#Preprocessing
print("Preprocessing")

tri=tr.shape[0]
y = tr.deal_probability.copy()

lb=LabelEncoder()

List_Var=['item_id', 'description', 'activation_date', 'image']


def Concat_Text(df,Columns,Name):
    df=df.copy()
    df.loc[:,Columns].fillna(" ",inplace=True)
    df[Name]=df[Columns[0]].astype('str')
    for col in Columns[1:]:
        df[Name]=df[Name]+' '+df[col].astype('str')
    return df
    
def Ratio_Words(df):
    df=df.copy()
    df['description']=df['description'].astype('str')
    df['num_words_description']=df['description'].apply(lambda x:len(x.split()))
    Unique_Words=df['description'].apply(lambda x: len(set(x.split())))
    df['Ratio_Words_description']=Unique_Words/df['num_words_description']
    return df
    
def Lenght_Columns(df,Columns):
    df=df.copy()
    Columns_Len=['len_'+s for s in Columns]
    for col in Columns:
        df[col]=df[col].astype('str')
    for x,y in zip(Columns,Columns_Len):
        df[y]=df[x].apply(len)
    return df    
    

tr_te=tr[tr.columns.difference(["deal_probability"])].append(te)\
     .pipe(Concat_Text,['city','param_1','param_2','param_3'],'txt1')\
     .pipe(Concat_Text,['title','description'],'txt2')\
     .pipe(Ratio_Words)\
     .pipe(Lenght_Columns,['title','description','param_1'])\
     .assign( category_name=lambda x: pd.Categorical(x['category_name']).codes,
              parent_category_name=lambda x:pd.Categorical(x['parent_category_name']).codes,
              region=lambda x:pd.Categorical(x['region']).codes,
              user_type=lambda x:pd.Categorical(x['user_type']).codes,
              param_1=lambda x:lb.fit_transform(x['param_1'].fillna('-1').astype('str')),
              param_2=lambda x:lb.fit_transform(x['param_2'].fillna('-1').astype('str')),
              param_3=lambda x:lb.fit_transform(x['param_3'].fillna('-1').astype('str')),
              user_id=lambda x:lb.fit_transform(x['user_id'].astype('str')),
              city=lambda x:lb.fit_transform(x['city'].astype('str')),
              price=lambda x: np.log1p(x['price'].fillna(0)),
             mon=lambda x: pd.to_datetime(x['activation_date']).dt.month,
             mday=lambda x: pd.to_datetime(x['activation_date']).dt.day,
             week=lambda x: pd.to_datetime(x['activation_date']).dt.week,
             wday=lambda x:pd.to_datetime(x['activation_date']).dt.dayofweek,
             title=lambda x: x['title'].astype('str'),
            image_top_1=lambda x:np.log1p(x['image_top_1'].fillna(0)))\
             .drop(labels=List_Var,axis=1)
             
tr_te.price.replace(to_replace=[np.inf, -np.inf,np.nan], value=-1,inplace=True)


del tr,te
gc.collect()

tr_te.loc[:,'txt2']=tr_te.txt2.apply(lambda x:x.lower().replace("[^[:alpha:]]"," ").replace("\\s+", " "))

print("Processing Text")
print("Text 1")

vec1=CountVectorizer(ngram_range=(1,2),dtype=np.uint8,min_df=5, binary=True,max_features=3000) 
m_tfidf1=vec1.fit_transform(tr_te.txt1)
tr_te.drop(labels=['txt1'],inplace=True,axis=1)

print("Text 2")

vec2=TfidfVectorizer(ngram_range=(1,2),stop_words=stopWords,min_df=3,max_df=0.4,sublinear_tf=True,norm='l2',max_features=5500,dtype=np.uint8)
m_tfidf2=vec2.fit_transform(tr_te.txt2)
tr_te.drop(labels=['txt2'],inplace=True,axis=1)

print("Title")
vec3=CountVectorizer(ngram_range=(3,6),analyzer='char_wb',dtype=np.uint8,min_df=5, binary=True,max_features=2000) 
m_tfidf3=vec3.fit_transform(tr_te.title)
tr_te.drop(labels=['title'],inplace=True,axis=1)

print("General Data")
data  = hstack((tr_te.values,m_tfidf1,m_tfidf2,m_tfidf3)).tocsr()

print(data.shape)
del tr_te,m_tfidf1,m_tfidf2,m_tfidf3
gc.collect()

dtest=data[tri:]
X=data[:tri]
print(X.shape)
del data
gc.collect()

#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=23)
#dtrain =lgb.Dataset(data = X_train, label = y_train)
#dval =lgb.Dataset(data = X_valid, label = y_valid)
#del(X, y, tri,X_train,X_valid,y_train,y_valid)
#gc.collect()

Dparam = {'objective' : 'regression',
          'boosting_type': 'gbdt',
          'metric' : 'rmse',
          'nthread' : 4,
          #'max_bin':350,
          'shrinkage_rate':0.03,
          'max_depth':18,
          'min_child_weight': 8,
          'bagging_fraction':0.75,
          'feature_fraction':0.75,
          'lambda_l1':0,
          'lambda_l2':0,
          'num_leaves':31}        

print("Training Model")

def RMSE(L,L1):
    return np.sqrt(mse(L,L1))
    
folds = KFold(n_splits=5, shuffle=True, random_state=50001)
oof_preds = np.zeros(X.shape[0])
sub_preds = np.zeros(dtest.shape[0])

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X)):
    dtrain =gbm.Dataset(X[trn_idx], y.iloc[trn_idx])
    dval =gbm.Dataset(X[val_idx], y.iloc[val_idx])
    m_gbm=gbm.train(params=Dparam,train_set=dtrain,num_boost_round=2000,verbose_eval=500,valid_sets=[dtrain,dval],valid_names=['train','valid'])
    oof_preds[val_idx] = m_gbm.predict(X[val_idx])
    sub_preds += m_gbm.predict(dtest) / folds.n_splits
    print('Fold %2d rmse : %.6f' % (n_fold + 1, RMSE(y.iloc[val_idx],oof_preds[val_idx])))
    del dtrain,dval
    gc.collect()
    
print('Full RMSE score %.6f' % RMSE(y, oof_preds))   

sub_preds[sub_preds<0]=0
sub_preds[sub_preds>1]=1

print("Output Model")
Submission=pd.read_csv("../input/sample_submission.csv")
Submission['deal_probability']=sub_preds
Submission.to_csv("baseline_lgb_Basev8.csv", index=False)
   