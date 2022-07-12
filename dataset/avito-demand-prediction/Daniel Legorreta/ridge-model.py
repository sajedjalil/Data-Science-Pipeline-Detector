#Basic Model
#Sparse Model 
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix, hstack
from multiprocessing import Pool
import gc
import time
import nltk
import re,string
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,HashingVectorizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('russian'))


print("="*50)
print("Load Data")
start_time = time.time()
train = pd.read_csv("../input/train.csv",parse_dates=['activation_date'])
test = pd.read_csv("../input/test.csv",parse_dates=['activation_date'])
test_id = test["item_id"].values
print("Size Data Train (%d,%d)"%(train.shape[0],train.shape[1]))
print("Size Data Test (%d,%d)"%(test.shape[0],test.shape[1]))



print("="*50)
print("Pre-Processesing")

def Create_Var_Price(df):
	L=df[['user_id','price']].groupby('user_id',as_index=False).agg({'price':[max,'mean','sum']})
	L.columns = ['user_id']+["_".join(x) for x in L.columns.ravel()][1:]
	return pd.merge(df,L,on='user_id',how='left')
	
text_cols1 = ['parent_category_name','category_name', 'param_1']
text_cols2 =['title','description']
num_features=['price','price_max','price_mean','price_sum']

def Create_Features(df):
    df['DayofMonth']=df.activation_date.dt.day
    df['Month']=df.activation_date.dt.month
    df['Week']=df.activation_date.dt.week
    df['Weekday']=df.activation_date.dt.weekday
    for c in num_features:
        df[c]=df[c].apply(lambda x: np.log1p(x))
        df[c].fillna(0,inplace=True)
    for col in text_cols1:
        df[col].fillna(" ",inplace=True)
    for col in text_cols2:
        df[col].fillna(" ",inplace=True)
    
    df['text_1'] = df.apply(lambda x: " ".join(x[col] for col in text_cols1), axis=1)
    df['text_2'] = df.apply(lambda x: " ".join(x[col] for col in text_cols2), axis=1)
    return df


print("Time Features")
train=Create_Var_Price(train)
test=Create_Var_Price(test)

print("Create Features")
train = Create_Features(train)
test = Create_Features(test)

train.drop(labels=text_cols1,axis=1,inplace=True)
test.drop(labels=text_cols1,axis=1,inplace=True)

train.drop(labels=text_cols2,axis=1,inplace=True)
test.drop(labels=text_cols2,axis=1,inplace=True)
gc.collect()

print("Size Data Train (%d,%d)"%(train.shape[0],train.shape[1]))
print("Size Data Test (%d,%d)"%(test.shape[0],test.shape[1]))

print("="*50)
print("Category Features")

lb=LabelEncoder()
Cat_dict={}
cat_features =['region','city','user_type','DayofMonth','Month','Week','Weekday','item_seq_number','image_top_1']

train.loc[:,cat_features].fillna(0.0,inplace=True)
test.loc[:,cat_features].fillna(0.0,inplace=True)
train.image_top_1.fillna(0,inplace=True)
test.image_top_1.fillna(0,inplace=True)

for col in cat_features:
    Cat_dict[col]=np.concatenate((train[col].unique(),test[col].unique()),axis=0)

for col in cat_features:
    lb.fit(Cat_dict[col])
    train[col]=lb.transform(train[col])
    test[col]=lb.transform(test[col])

mean_dc=dict()
for feat in cat_features:
    mean_dc[feat] = train.groupby(feat)['deal_probability'].mean().astype(np.float32)
    mean_dc[feat] /= np.max(mean_dc[feat])
    train['mean_deal_'+feat] = train[feat].map(mean_dc[feat]).astype(np.float32)
    train['mean_deal_'+feat].fillna( mean_dc[feat].mean(), inplace=True  )
    
for feat in cat_features:
    test['mean_deal_'+feat] = test[feat].map(mean_dc[feat]).astype(np.float32)
    test['mean_deal_'+feat].fillna( mean_dc[feat].mean(), inplace=True  )



print("Size Data Train (%d,%d)"%(train.shape[0],train.shape[1]))
print("Size Data Test (%d,%d)"%(test.shape[0],test.shape[1]))

train.drop(labels=cat_features,axis=1,inplace=True)
test.drop(labels=cat_features,axis=1,inplace=True)
cat_features=['mean_deal_'+feat for feat in cat_features]
print("Size Data Train (%d,%d)"%(train.shape[0],train.shape[1]))
print("Size Data Test (%d,%d)"%(test.shape[0],test.shape[1]))
X_train_cat =csr_matrix(train[cat_features])
X_test_cat = csr_matrix(test[cat_features])
train.drop(labels=cat_features,axis=1,inplace=True)
test.drop(labels=cat_features,axis=1,inplace=True)
#csr cat_features

del Cat_dict,mean_dc
gc.collect()

print("="*50)
print("Num Features")

scaler = StandardScaler()
X_train_num =csr_matrix( scaler.fit_transform(train[num_features]))
X_test_num = csr_matrix(scaler.transform(test[num_features]))

#X_train_num=csr_matrix(X_train_num)
#X_test_num=csr_matrix(X_test_num)

train.drop(labels=num_features,axis=1,inplace=True)
test.drop(labels=num_features,axis=1,inplace=True)

print("="*50)
print("Text Features")

def Clean_Basic(text):
    pattern1 = re.compile(r'\n\t\r')
    punt = re.compile('[%s]' % re.escape(string.punctuation))
    text=str(text).lower()
    text=re.sub(u"\/",u"",text)
    text=re.sub(u"\\n",u" ",text)
    text=re.sub(pattern1, '', text)
    text=punt.sub('',text)
    return text

cores=4
def parallelize_dataframe(df, func):
    df_split = np.array_split(df, cores)
    pool = Pool(cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
def clean_str_df(df):
    return df.apply( lambda s : Clean_Basic(str(s)))

train["text_1"]=parallelize_dataframe(train["text_1"],clean_str_df)
test["text_1"]=parallelize_dataframe(test["text_1"],clean_str_df)

train["text_2"]=parallelize_dataframe(train["text_2"],clean_str_df)
test["text_2"]=parallelize_dataframe(test["text_2"],clean_str_df)

def create_count_features(df_data):
    def lg(text):
        text = [x for x in text.split() if x!='']
        return len(text)
    #df_data['nb_words_text1'] = df_data['text_1'].apply(lg).astype(np.uint16)
    return df_data.apply(lg).astype(np.uint16)
    
train["num_words_text_1"]=parallelize_dataframe(train["text_1"],create_count_features)
test["num_words_text_1"]=parallelize_dataframe(test["text_1"],create_count_features)

train["num_words_text_2"]=parallelize_dataframe(train["text_2"],create_count_features)
test["num_words_text_2"]=parallelize_dataframe(test["text_2"],create_count_features)

vec_text1=HashingVectorizer(ngram_range=(1,2),stop_words=stop_words,binary=True,dtype=np.uint8,n_features=2**9)
vec_text2=HashingVectorizer(ngram_range=(1,3),stop_words=stop_words,binary=True,dtype=np.uint8,n_features=2**16)
#(ngram_range=(1,3),stop_words=stop_words,min_df=5,max_df=0.3,max_features=5000,binary=True,dtype=np.uint8)
Text_1=pd.concat([train['text_1'],test['text_1']])
Text_2=pd.concat([train['text_2'],test['text_2']])

text1_tfidf=vec_text1.fit_transform(Text_1)
text2_tfidf=vec_text2.fit_transform(Text_2)

tr=train.shape[0]
print(text1_tfidf.shape)
print(text2_tfidf.shape)

print("="*50)
print("General")

X_Target=train.deal_probability.copy()

General_Train = hstack((train[["num_words_text_1","num_words_text_2"]].values,X_train_num,text1_tfidf[:tr],text2_tfidf[:tr])).tocsr()
print(General_Train.shape)

General_Test = hstack((test[["num_words_text_1","num_words_text_2"]].values,X_test_num,text1_tfidf[tr:],text2_tfidf[tr:])).tocsr()
print(General_Test.shape)

del X_train_num,X_test_num,text1_tfidf,text2_tfidf
gc.collect()

print("="*50)
print("Train Model")
model= Ridge(alpha=10, copy_X=True, fit_intercept=True, solver='auto',
                    max_iter=500,   normalize=False, random_state=0,  tol=0.0025)
                    
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, model_selection, metrics

# Splitting the data for model training#
X_train, X_test, y_train, y_test = train_test_split(General_Train, X_Target, test_size=0.30, random_state=42)
    
print(X_train.shape, X_test.shape,y_train.shape,y_test.shape)
print(General_Train.shape)

model.fit(X_train,y_train)
Y_Pred=model.predict(X_test)
print(np.sqrt(metrics.regression.mean_squared_error(y_test,Y_Pred)))
pred_test=model.predict(General_Test)

pred_test[pred_test<0]=0
pred_test[pred_test>1]=1
# Making a submission file #
sub_df = pd.DataFrame({"item_id":test_id})
sub_df["deal_probability"] = pred_test
sub_df.to_csv("baseline_Ridge_Regression.csv", index=False)