#-*- coding: utf-8 -*-
"""
Created on Thu Nov 30 15:51:51 2017

@author: Yury
"""
kag=1

import time
#from sklearn import preprocessing
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#from wordcloud import WordCloud
#from sklearn.feature_extraction.text import TfidfVectorizer
#import string
import re
import gc
import sys 

from fastcache import clru_cache 

import scipy.sparse as sp
gc.collect()

import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

'''models to use'''
ridg=0
lgbm=0
ann=1

hm=0 #if 1 calculate metric for train
@clru_cache(1024)
def tokenize(text):
    """
    sent_tokenize(): segment text into sentences
    word_tokenize(): break sentences into words
    """
    try: 
        regex = re.compile('[' +re.escape(string.punctuation) + '\\r\\t\\n]')
        text = regex.sub(" ", text) # remove punctuation
        
        tokens_ = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(text)]
        tokens = []
        stop = set(stopwords.words('english'))
        for token_by_sent in tokens_:
            tokens += token_by_sent
#        tokens = list(filter(lambda t: t.lower() not in stop, tokens))
        #tokens = list(filter(lambda t: t.lower()))
        filtered_tokens = [w.lower() for w in tokens if re.search('[a-zA-Z0-9]', w)]
        filtered_tokens = [w for w in filtered_tokens if len(w)>=3] #>=3 letters in a word
        
        return filtered_tokens
            
    except TypeError as e: print(text,e)

start = time.clock()

#Error calculation
# vectorized error calc
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))


def memory_reduce(dataset):
    for col in list(dataset.select_dtypes(include=['int']).columns):
        if ((np.max(dataset[col]) <= 127) and(np.min(dataset[col]) >= -128)):
            dataset[col] = dataset[col].astype(np.int8)
        elif ((np.max(dataset[col]) <= 32767) and(np.min(dataset[col]) >= -32768)):
            dataset[col] = dataset[col].astype(np.int16)
        elif ((np.max(dataset[col]) <= 2147483647) and(np.min(dataset[col]) >= -2147483648)):
            dataset[col] = dataset[col].astype(np.int32)
    for col in list(dataset.select_dtypes(include=['float']).columns):
        dataset[col] = dataset[col].astype(np.float32)
    return dataset

def col_to_category(df, col):
    df[col] = df[col].astype('category')
    return df

    

def fill_missing_brand(original_df, column_name, replacements, inplace=True):
    df = original_df if inplace else original_df.copy()
    for word in replacements:
        empty = pd.isnull(df[column_name])
        if not empty.any():
            return df
        contained = (df.loc[empty, "name"].str.contains(word))  | (df.loc[empty, 'item_description'].str.contains(word))
        df.loc[contained[contained].index, column_name] = word
    df.brand_name.fillna(value="missing", inplace=True)
    df.loc[~df['brand_name'].isin(replacements), 'brand_name'] = 'missing'
    return df

def handle_missing(dataset):
    dataset.category_name.fillna(value="missing", inplace=True)
    dataset.item_description.fillna(value="missing", inplace=True)
    return (dataset)

def unique_brand(dataset, col_name, thresh=0.0003):
    unique, counts = np.unique(dataset.loc[dataset[col_name].notnull()]["brand_name"], return_counts=True)
    a=pd.DataFrame({'val':unique, 
                    'freq':counts}).sort_values(['freq'], ascending=False)
    #frequent brands
    #print('freq threshold: ', thresh*train.shape[0])
    a=a.loc[(a['freq']>thresh*dataset.shape[0]) & (a['val']!='missing')]['val'].tolist()
    # missing will be raplace with longer and not missing then
    a=sorted(a,key = len, reverse=True)
    return a

def transform_cat_name(category_name):
    try:
        main, sub1, sub2= category_name.split('/')
        return main, sub1, sub2
    except:
        return 'missing', 'missing', 'missing'

def miss_val(dataset, col_name):
    dataset[col_name].fillna(value="missing", inplace=True)


def corr_study(df, col, word, crl=0.05):
    df["0_"+col+"_"+word]=0
    df.loc[df[col].str.count(word)>0, col+"_"+word] =1
    cr=df['price'].corr(df[col+"_"+word])
    if abs(cr)<crl or np.isnan(cr):
#        print('drop')
        df.drop([col+"_"+word], axis=1)  
    else:
        print("correlation ", col, " with ", word, " is ",  cr)
    return df 

def rmsle_study(df, col, word, crl=0.05):
    cr=rmsle(df['price'], df[col+'_'+word])
    if abs(cr)>crl or (~np.isnan(cr)):
        print("rmsle ", col, " with ", word, " is ",  cr)
    return df 


def model_predict(regressor,xtrain, ytrain, xtest):
    regr = regressor
    regr.fit(xtrain,ytrain) 
    tr=regr.predict(xtrain).astype(np.float32)
    te=regr.predict(xtest).astype(np.float32)
    del regr
    return tr, te

def map_column(df, dic, col, suf):
    df["0_"+col+suf] = df[col].map(dic).fillna("missing")
    return df



def panda_spicy(df):
    row_index=df.index
    col_name=list(df.columns)
    df=sp.csr_matrix(df)
    return df, row_index, col_name


def sparse_pandas(df, row, col):
    df=pd.DataFrame(df.todense()) #no direct way
    df.index=row #keep index from pandas df
    df.columns=col #set column names from pandas df
    return df

#def main():


#------------------------------------------------
if kag==0:    
    #train =pd.read_csv("train.tsv", sep="\t",  engine="python",encoding='utf-8')
    #test = pd.read_csv("test.tsv", sep="\t",  engine="python",encoding='utf-8')
    #train=train[:50000]
    #test=test[:50000]
    ##
    #test.to_csv("test_short.csv", index = False)
    #train.to_csv("train_short.csv", index = False)
    
    train =pd.read_csv("train_short.csv",  engine="python")
    test = pd.read_csv("test_short.csv",  engine="python")
else:

    #train =pd.read_csv("../input/train.tsv", sep="\t",  engine="python")
    #test = pd.read_csv("../input/test.tsv", sep="\t",  engine="python")
    
    train =pd.read_csv("../input/train.tsv", sep="\t",  engine="c")
    test = pd.read_csv("../input/test.tsv", sep="\t",  engine="c")
start_mem_tr=train.memory_usage().sum()

train_ex=pd.DataFrame()
test_ex=pd.DataFrame()

#_____________________________________
add_dummies=1
nlp_yes=0
#_____________________________________________
memory_reduce(train)
memory_reduce(test)


gc.collect()


print('memory change: ',(train.memory_usage().sum()-start_mem_tr)/start_mem_tr)
start_mem_tr=train.memory_usage().sum()
print('memory train: ', start_mem_tr)
#-----TARGET------------------------------------------------
col='price'
train[col].describe()
train[col].value_counts()
unique, counts = np.unique(train['price'], return_counts=True)
a=pd.DataFrame({'val':unique, 
                'freq':counts}).sort_values(['val'], ascending=False)
#a.plot()    

print(train['price'].groupby(train['item_condition_id']).quantile(.98))
print(train['price'].quantile(.099))
print(train['price'].groupby(train['item_condition_id']).std())

dic={1:1, 2:1, 3:1, 4:2, 5:2}
map_column(train, dic, 'item_condition_id',"_gen")
map_column(test, dic, 'item_condition_id',"_gen")

a=train[['name', 'brand_name','item_description','price']].sort_values(['price'], ascending=False)
 
#outliers
col='price'
train=train.loc[train[col]>=3] 
av_price=np.percentile(train['price'], 99)
train=train[train[col]<av_price]  

#train.reset_index(drop=True, inplace=True)

#log+1 transform
train[col] = np.log1p(train[col])

mean = train[col].mean()
std = train[col].std()
mean=0
std=1
train[col] = (train[col] - mean) / std

#y = y.reshape(-1, 1)
#----------------------------------



#X_train=pd.DataFrame()
#X_test=pd.DataFrame()


#___-----------------------------------------------------
#train_sp=pd.DataFrame()
#test_sp=pd.DataFrame()

train_id=train['train_id']
#train_sp['train_id']=train_id


test_id=test['test_id']
#test_sp['test_id']=test_id


#train_sp['cond_id']=train['item_condition_id']
#test_sp['cond_id']=test['item_condition_id']
#train_sp['shipping']=train['shipping']
#test_sp['shipping']=test['shipping']



#_____________________________________
#test.columns
#test = pd.read_table("test.tsv")
#import scipy as sp
#test = sp.genfromtxt("test.tsv", delimiter="\t",encoding='utf-8')

#train=train[[col for col in train.columns if col not in ['price']] + ["price"]]
#




#________________________________________________________
#train.head(10)
#Line #692665 (got 2 columns instead of 7)

#a=test.loc[test['test_id'].isin( [692665,692666,692663,692664])]

print("Handling missing values...")



t = time.clock()
train = handle_missing(train)
test = handle_missing(test)
print("handled in ", (time.clock()-t))

t=time.clock()
col='item_description'
train[col]=train[col].replace('No description yet', 'missing')
test[col]=test[col].replace('No description yet', 'missing')

col="category_name"
train[col]=train[col].map(lambda x: re.sub("[?&+,-]", "", x))
test[col]=test[col].map(lambda x: re.sub("[?&+,-]", "", x))

#Replace e'
for col in ['name','item_description', "brand_name", 'category_name']:
    train[col]=train[col].str.lower()
    train[col]=train[col].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    #
    test[col]=test[col].str.lower()
    test[col]=test[col].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
#train.head()


#for col in ["brand_name", "item_condition_id", "category_name", "shipping" ]:
#    col_to_category(train,col)
#    col_to_category(test,col)
    
    
print("replace in ", (time.clock()-t))
#price_brand_med=train["price"].groupby(train[col]).median().sort_values(ascending=False)

#memo
gc.collect()

#____________________________________________________________________

#df['col'] = ['new string' if '@' in x else x for x in df['col']]

#train['item_condition_id'].value_counts()
#price_av={}
#price_med={}
#for i in range(1,6):
#    price_av[i]=train['price'].loc[train['item_condition_id']==i].mean()
#    price_med[i]=train['price'].loc[train['item_condition_id']==i].median()

#_______________________________________
#col='category_name'
#col2='item_condition_id'
#price_av_cat={}
#price_med_cat={}
#for i in train[col]:
#    for j in train[col2]:
#        price_av_cat[i]=train['price'].loc[train[col2]==j].mean()
#        price_med_cat[i]=train['price'].loc[train[col2]==j].median()

#groupby
#price_cat_mean=train["price"].groupby(train[col]).mean() 
#price_cat_med=train["price"].groupby(train[col]).median()
#
##cat+cond->med price
#train['cat_cond']=train['category_name'].astype(str)+train['item_condition_id'].astype(str)
#test['cat_cond']=test['category_name'].astype(str)+test['item_condition_id'].astype(str)
#price_cat_qual_med=train["price"].groupby(train['cat_cond']).median().sort_values(ascending=False).to_dict()
#
#train['pr_cat_con_med']=train['cat_cond'].map(price_cat_qual_med).astype(int)
#test['pr_cat_con_med']=test['cat_cond'].map(price_cat_qual_med).astype(int)

#___________

#brand+condition


#_________________!!!!!!!!!!!!!!!!!!_______________________________________________________-      
t=time.clock()
#Brands in datasets
#a=list(set(train["brand_name"].unique()))
#a.remove('missing')
#
#print(train["brand_name"].value_counts()  )




a=unique_brand(train, "brand_name", 0.0003)
print ('A list has :', len(a))

#b=list(set(test["brand_name"].unique()))
#c=list(set(a+b))

#for word in a:
##    test.loc[(test["brand_name"]=='missing') & ((test["name"].str.count(word)>0) | (test['item_description'].str.count(word)>0)), "brand_name"]=word
##    train.loc[(train["brand_name"]=='missing') & ((train["name"].str.count(word)>0)  | (train['item_description'].str.count(word)>0)), "brand_name"]=word
#    test.loc[(test["brand_name"]=='missing') & ((test["name"].str.contains(word)) | (test['item_description'].str.contains(word))), "brand_name"]=word
#    train.loc[(train["brand_name"]=='missing') & ((train["name"].str.contains(word))  | (train['item_description'].str.contains(word))), "brand_name"]=word



fill_missing_brand(train, "brand_name", a, True)
fill_missing_brand(test, "brand_name", a, True)

#cut number of brands
#train.loc[~train['brand_name'].isin(a), 'brand_name'] = 'missing'
#test.loc[~test['brand_name'].isin(a), 'brand_name'] = 'missing'


#print(train["brand_name"].value_counts()  )



print("brand ", (time.clock()-t))
print("brand since start ", (time.clock()-start))
#_________________!!!!!!!!!!!!!!!!!!_______________________________________________________-      

extra_col=[]
extra_cat=["category_name", 'brand_name']
'''
/////////////////////////////////////////////////////////
#--------Split category---------------------------------------------
////////////////////////////////////////////////////////
'''

#add columns category split
t=time.clock()

train['cat_main'], train['cat_sub1'], train['cat_sub2'] = zip(*train['category_name'].apply(transform_cat_name))
test['cat_main'], test['cat_sub1'], test['cat_sub2'] = zip(*test["category_name"].apply(transform_cat_name))

extra_cat.extend(['cat_main','cat_sub1','cat_sub2'])

#missing category


for col in ['cat_main','cat_sub1','cat_sub2']:
    miss_val(test, col)
    miss_val(train, col)
#----+++++_____
    



#__Level_1_______________________________________
col='cat_main'
main_categories = [c for c in train[col].unique() if type(c)==str]
categories_sum=0
dic={}
for c in main_categories:
    b=100*len(train[train[col]==c])/len(train)
    if b>3:
        dic[c]=c
    else:
        dic[c]="missing"
    categories_sum+=b
#    print('{:25}{:3f}% of training data'.format(c, b))
#print('nan\t\t\t {:3f}% of training data'.format(100-categories_sum))
    
#map generilized categories
map_column(train, dic, col, "_gen")
map_column(test, dic, col, "_gen")
extra_cat.append("0_"+col+ "_gen")  

#____Level_2_____________________________________
#main+cat_1
col='cat_m_s1'
train[col]=train['cat_main'].astype(str)+" " +train['cat_sub1'].astype(str)
test[col]=test['cat_main'].astype(str)+" " +test['cat_sub1'].astype(str)
extra_cat.append(col)

main_categories = [c for c in train[col].unique() if type(c)==str]
categories_sum=0
dic={}
for c in main_categories:
    b=100*len(train[train[col]==c])/len(train)
    if b>1:
        dic[c]=c
        #print('{:25}{:3f}% of training data'.format(c, b))
    else:
        dic[c]="missing"
    categories_sum+=b
    #print('{:25}{:3f}% of training data'.format(c, b))
#print('nan\t\t\t {:3f}% of training data'.format(100-categories_sum))

#map generilized categories
map_column(train, dic, col, "_gen")
map_column(test, dic, col, "_gen")
extra_cat.append("0_"+col+ "_gen")   

#___Level_3______________________________________
col='category_name'
main_categories = [c for c in train[col].unique() if type(c)==str]
categories_sum=0
dic={}
for c in main_categories:
    b=100*len(train[train[col]==c])/len(train)
    if b>1:
        dic[c]=c
        #print('{:25}{:3f}% of training data'.format(c, b))
    else:
        dic[c]="missing"
    categories_sum+=b
    #print('{:25}{:3f}% of training data'.format(c, b))
#print('nan\t\t\t {:3f}% of training data'.format(100-categories_sum))

#map generilized categories
map_column(train, dic, col, "_gen")
map_column(test, dic, col, "_gen")

#_________________________________________
 
print("split category", (time.clock()-t))

'''
//////////////////////////////////////////////////////////
            DUMMIES
//////////////////////////////////////////////////////////
'''
#_________________________________________________________________
if add_dummies==1:
    t2=time.clock()
    col_dum=[]
    train_dum=sp.csr_matrix(pd.DataFrame())
    test_dum=sp.csr_matrix(pd.DataFrame())
    
    def get_dum_sparse(train, test, col, train_dum, test_dum, col_dum):
        col_d=[]
        train_d=pd.DataFrame()
        test_d=pd.DataFrame()
        #pd.get_dummies(train.loc[:, columns_for_pc], sparse=True)
        train_d = pd.get_dummies(train[col].astype(str)).astype(np.int8).to_sparse(fill_value=0)       
        test_d = pd.get_dummies(test[col].astype(str)).astype(np.int8).to_sparse(fill_value=0)
        #re-index the new data to the columns of the training data, filling the missing values with 0
        test_d=test_d.reindex(columns = train_d.columns) #, fill_value=0
        col_d=[s+"_"+col for s in train_d.columns]
        
        train_d=sp.csr_matrix(train_d)     
        test_d=sp.csr_matrix(test_d) 
        col_dum.extend(col_d)
        
        train_dum=sp.hstack((train_dum,train_d))
        test_dum=sp.hstack((test_dum,test_d))
        return train_dum, test_dum, col_dum
    
    
    for col in ['shipping','item_condition_id', '0_item_condition_id_gen']: #, "brand_name",'cat_main','cat_sub1','cat_sub2']:
        train_dum, test_dum, col_dum=get_dum_sparse(train, test, col, train_dum, test_dum, col_dum)



#        # Feature Scaling
#        from sklearn.preprocessing import StandardScaler
#        sc = StandardScaler()
#        X_train = sc.fit_transform(X_train)
#        X_test = sc.transform(X_test)
        
        # Applying Kernel PCA
#        from sklearn.decomposition import KernelPCA
#        n=5
#        kpca = KernelPCA(n_components = n, kernel = 'rbf') #, n_jobs=-1)
#        
#        train_X = kpca.fit_transform(train_X)
#        test_X = kpca.transform(test_X)
#       
        
#        X_train=pd.DataFrame(X_train).add_suffix("_"+str(l))
#        X_test=pd.DataFrame(X_test).add_suffix("_"+str(l))
#        
#        train_dum.index=train.index
#        test_dum.index=test.index
#        train=pd.concat([train, X_train], axis=1)
#        test=pd.concat([test,X_test], axis=1)
        
        print("dum2",train.shape)
#    del train_X
#    del test_X
    #print(train_X.info())
    print('Size of dataframe', sys.getsizeof(train_dum))
    t2=(time.clock()-t2)
    print("dummies ", (t2))
    

#------memo---------------------
gc.collect()
#start_mem=train.memory_usage().sum()+test.memory_usage().sum() ###+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
#memory_reduce(train)
#memory_reduce(test)
##memory_reduce(train_sp)
##memory_reduce(test_sp)
#end_mem=train.memory_usage().sum()+test.memory_usage().sum() ###+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
#print('memory change: ',(end_mem-start_mem)/start_mem)

#-------------------------------
t=time.clock()



#cat_main+brand
col='catm_brand'
train['catm_brand']=train['cat_main'].astype(str)+train['brand_name'].astype(str)
test['catm_brand']=test['cat_main'].astype(str)+test['brand_name'].astype(str)
extra_cat.append(col)
#lb = LabelBinarizer(sparse_output=True)
#X_brand = lb.fit_transform(merge['brand_name'])
#print('[{}] Finished label binarize `brand_name`'.format(time.time() - start_time))
#
#X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
#                                          sparse=True).values)
#    
more=0
if more==1:   
    #cat_1+brand
    col='cat1_brand'
    train[col]=train['cat_sub1'].astype(str)+train['brand_name'].astype(str)
    test[col]=test['cat_sub1'].astype(str)+test['brand_name'].astype(str)
    extra_cat.append(col)
    
    #main+cat_1+brand
    col='cat_m_2_brand'
    train[col]=train['cat_main'].astype(str)+train['cat_sub1'].astype(str)+train['brand_name'].astype(str)
    test[col]=test['cat_main'].astype(str) + test['cat_sub1'].astype(str)+test['brand_name'].astype(str)
    extra_cat.append(col)
    
    #cat_2+brand
    col='cat2_brand'
    train[col]=train['cat_sub2'].astype(str)+train['brand_name'].astype(str)
    test[col]=test['cat_sub2'].astype(str)+test['brand_name'].astype(str)
    extra_cat.append(col)
    
    #cat_1+brand+qual
    col='cat1_brand_qu'
    train[col]=train['cat_sub1'].astype(str)+train['brand_name'].astype(str)+train['item_condition_id'].astype(str)
    test[col]=test['cat_sub1'].astype(str)+test['brand_name'].astype(str)+test['item_condition_id'].astype(str)
    extra_cat.append(col)
    
    #cat_2+brand+qual
    col='cat2_brand_qu'
    train[col]=train['cat_sub2'].astype(str)+train['brand_name'].astype(str)+train['item_condition_id'].astype(str)
    test[col]=test['cat_sub2'].astype(str)+test['brand_name'].astype(str)+test['item_condition_id'].astype(str)
    extra_cat.append(col)

    #"category_name"+brand+qu
    col='full_cat_brand_ship'
    train[col]=train["category_name"].astype(str)+train['brand_name'].astype(str)+train['shipping'].astype(str)
    test[col]=test["category_name"].astype(str)+test['brand_name'].astype(str)+test['shipping'].astype(str)
    extra_cat.append(col)
    
#"category_name"+brand+qu
col='full_cat_brand'
train[col]=train["category_name"].astype(str)+train['brand_name'].astype(str)+train['item_condition_id'].astype(str)
test[col]=test["category_name"].astype(str)+test['brand_name'].astype(str)+test['item_condition_id'].astype(str)
extra_cat.append(col)



print('cat_brand combinations %.3f seconds ' %(time.clock() -t) ) 




#------memo---------------------
gc.collect()
#start_mem=train.memory_usage().sum()+test.memory_usage().sum()##+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
#memory_reduce(train)
#memory_reduce(test)
#
#end_mem=train.memory_usage().sum()+test.memory_usage().sum()##+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
#print('memory change: ',(end_mem-start_mem)/start_mem)
#-------------------------------


#Brand-> top percentile
#Brand-> low percentile

    
#In test there are names that are not in training set
#train["category_name"].value_counts().head(10)
#train["brand_name"].value_counts().head(10)

#
#condition - good - bad - ex... + median price

#shipping -> median price

#has brand / description
t=time.clock()







def emb_col_0(tr, te,  tr_ex, te_ex, col, wo, suf):
    nn="0_"+col+suf
    extra_col.append(nn)
    pat = '|'.join([r'\b{}\b'.format(x.strip()) for x in wo])
    #pat = '|'.join(wo)
#    tr[nn] = np.where(tr[col].str.contains(pat, case=False, na=False).astype(int),0,1)
#    te[nn] = np.where(te[col].str.contains(pat, case=False, na=False).astype(int),0,1)
    tr_ex[nn] = 1-tr[col].str.contains(pat, case=False, na=False).astype(int)
    te_ex[nn] = 1-te[col].str.contains(pat, case=False, na=False).astype(int)
    return tr_ex, te_ex

def emb_col_1(tr, te, tr_ex, te_ex, col, wo, suf):
    nn="0_"+col+suf
    extra_col.append(nn)
    pat = '|'.join([r'\b{}\b'.format(x.strip()) for x in wo])
    #pat = '|'.join(wo)
    tr_ex[nn] = tr[col].str.contains(pat, case=False, na=False).astype(int)
    te_ex[nn] = te[col].str.contains(pat, case=False, na=False).astype(int)
    return tr_ex, te_ex

col_s=["name",'item_description']



words=['missing']
for col in ["brand_name",'item_description']:
    emb_col_0(train, test, train_ex, test_ex, col, words, '_yes')
    
    
    
#train['HasDescription']=(train['item_description']!='missing')*1
#test['HasDescription']=(test['item_description']!='missing')*1
  
    
#has brand in descr
col="brand_name"
col2='item_description'
test_ex['0_hasBrandDesc']=(test.apply(axis = 1, func = lambda x: str(x[col]).lower() in str(x[col2])))*1
train_ex['0_hasBrandDesc']=(train.apply(axis = 1, func = lambda x: str(x[col]).lower() in str(x[col2])))*1

extra_col.append('0_hasBrandDesc')


#most expensive brands


#most expensive categories

embed=1
if embed==1:
    
    words2replace=[]
    #with photo
    # Add whether it may have pictures
    #pic_word_re = [re.compile(r, re.IGNORECASE) for r in [r'(see(n)?)?( in| the| my) (picture(s)?|photo(s)?)']]
    words=['photo','picture', 'pic', 'photos', 'pictures', ' pics ', 'pix', 'image', 'images', 
           'photograph', 'can see', 'can look', 'can view', 'may see', 
           'may look', 'may view', 'see all', 'see pics',
           'see_what', 'see through', 'see the' ,'see pic', 'see other', 'see for',
           'shown', 'shown picture', 'shown pictures']
    words2replace+=words
    suf='_pic'
    for col in col_s:
        emb_col_1(train, test, train_ex, test_ex, col, words, suf)

embed=1
if embed==1:
    #several items
    words=['pairs', 'bundle', 'bundles','set', 'sets',  'pcs', 'peaces', 'kit', 'batch', 
           'collection', 'pack', 'band', 'group', 'items', 'array',
           'assortment', 'each', 'includes', 'comes with', 'all in', 'per pc', 'lot',
           'listing', 'per', 'see all'
           ]
    words2replace+=words
    suf='_set'
    for col in col_s:
        emb_col_1(train, test, train_ex, test_ex, col, words, suf)
        
    
    #contains [rm]
embed=1
if embed==1:        
    suf='_rm'
    for col in col_s:
        emb_col_1(train, test, train_ex, test_ex, col, '\[rm\]', suf)
    #    cols="0_"+col+suf
    #    extra_col.append(cols)
    #    test[cols]=0
    #    test.loc[test[col].str.count('\[rm\]')>0, cols] =1
    #    
    #    
    #    train[cols]=0
    #    train.loc[train[col].str.count('\[rm\]')>0, cols] =1
        
embed=0
if embed==1:    
    #brand new without tags
    words=["bnib", "bnwt", 'b.n.w.t.',
              "bnwot", 'b.n.w.o.t.', "refurbished", 
              'like new', 'worn', 'times', 'condition', 'cleaning', 'once', 'twice',
              'used but still',
              'used but good', 'wear',	'tear',	'no key', 'flaw', 'flaws', 'tears'
              ]
    words2replace+=words
    suf='_bnwt'
    for col in col_s:
        emb_col_1(train, test, train_ex, test_ex, col, words, suf)
        
    
    #new
    words=['nwt', "N.W.T.", 'n.w.t.', "new with tag", 'unused', 'never used', 'never opened', 
         'new in box', 'unopened', 'sealed']
    words2replace+=words
    suf='_new'
    for col in col_s:
        emb_col_1(train, test, train_ex, test_ex, col, words, suf)
embed=1
if embed==1:       
    #karat gold
    words=['10k', '12k', '14k', '16k', '18k', '20k', '22k', '24k', '26k', '28k', '30k', '32k', '34k', '36k', '48k', 
           '10kt', '12kt', '14kt', '16kt', '18kt', '20kt', '22kt', '24kt', '26kt', '28kt', '30kt', '32kt', '34kt', '36kt', '48kt',
           '10 karat', '12 karat', '14 karat', '16 karat', '18 karat', '20 karat', '22 karat', '24 karat', '26 karat', '28 karat', '30 karat', '32 karat', '34 karat', '36 karat', '48 karat', 
           'gold', 'platinum',	'silver', 'diamond']	
    words2replace+=words
    suf='_karat'
    for col in col_s:
        emb_col_1(train, test, train_ex, test_ex, col, words, suf)

embed=0
if embed==1:    	
    #reserved
    words=['reserved', 'on hold', 'hold']
    words2replace+=words
    suf='_hold'
    for col in col_s:
        emb_col_1(train, test, train_ex, test_ex, col, words, suf)
    
    
    #china
    words=['made in china', 'cheap', 'low', 'bad', 'faux']
    words2replace+=words
    suf='_cheap'
    for col in col_s:
        emb_col_1(train, test, train_ex, test_ex, col, words, suf)

embed=1
if embed==1:    
    #
    words=['case',	'cover', 'charger']
    words2replace+=words
    suf='_case'
    for col in col_s:
        emb_col_1(train, test, train_ex, test_ex, col, words, suf)
        
    #'insert', 'dust'
    words=['insert', 'dust']
    words2replace+=words
    suf='_ins'
    for col in col_s:
        emb_col_1(train, test, train_ex, test_ex, col, words, suf)
    #
    words=['like','looks like','look-a-like','look a like', 'no brand', 'fake', 'imitating', 
           'imitation']
    words2replace+=words
    suf='_like'
    for col in col_s:
        emb_col_1(train, test, train_ex, test_ex, col, words, suf)
    

    
    
    #shipping
    words=['ship within', 'shipping cost', 'save shipping'
            'shipped with', 
            'ship with', 
            'shipping and', 
            'ship the', 
            'ship same', 
            'ship out', 
            'shipping costs',
            'shipping included',
            'price shipping', 
            'shipping you',
            'shipping with',
            'shipping will',
            'shipping price',
            'shipping']
    words2replace+=words
    suf='_ship'
    for col in col_s:
        emb_col_1(train, test, train_ex, test_ex, col, words, suf)

embed=1
if embed==1:    
    ##
    words=['100%','excellent', 'genuine', 'gorgeous', 'vintage', 'retro', 'perfect', 'unique', 'antique set', 'solid gold', 'edition', 'limited edition', 'original']
    words2replace+=words
    suf='_high'
    for col in col_s:
        emb_col_1(train, test, train_ex, test_ex, col, words, suf)
    
    ##
    words=['protect',
            'protection',
            'protective',
            'protector',
            'protects'
            ]
    words2replace+=words
    suf='_protect'
    for col in col_s:
        emb_col_1(train, test, train_ex, test_ex, col, words, suf)
     
    ##
    words=['ps2',
            'ps3',
            'ps4',
            ]
    words2replace+=words
    suf='_ps3'
    for col in col_s:
        emb_col_1(train, test, train_ex, test_ex, col, words, suf)
        
    #____________________________________________

embed=1
if embed==1:    
    words=['authentic',
            'perfume',
            'iphone',
           'please'
            'bnwt',
            'box',
            'size',
            'comes',
            'condition',
            'free',
            'hardware',
            'htf',
            
            'new',
            'nwt',
            'original',
            'paid',
            'price',
            'rare',
            'retail',
            'save',
            'signature',
            'sticker',
            'tags',
            'tracking',
            'worn',
            'zip',
            'not',
            'non',
            'prom'
            
            ]
    suf="_word"
    words2replace+=words
    for word in words[:3]:
        for col in col_s:
            emb_col_1(train, test, train_ex, test_ex, col, word, "_"+word)
    
    
    
    
    #for col in col_s:
    #    for word in words:
    #        rmsle_study(train, col, word, 0.05)
    
    
    #for col in col_s:
    #    for word in words2replace:
    #        corr_study(train, col, word, 0.05)   
    
    #plt.scatter(train['price'],train['item_description_auth'])
    #print("correlation ",  train['price'].corr(train['item_description_N.W.T.']))
    #
    #______________________________________
    #join and clear name and descr
    #    col='name_desc'
    #    test[col]=test['name'].astype(str)+(". ")+test['item_description'].astype(str)
    #    train[col]=train['name'].astype(str)+(". ")+train['item_description'].astype(str)
    #test[col]=test['item_description'].astype(str)
    #train[col]=train['item_description'].astype(str)
    
    
    ##replace symbols
    #train[col] = train[col].map(lambda x: re.sub('[^a-zA-Z ]', '', x))
    #test[col] = test[col].map(lambda x: re.sub('[^a-zA-Z ]', '', x))
    #
    #
    ##Tokenize???? split(' ') and clear duplicates
    #
    #train[col]=train[col].apply(lambda x: list( set(x.split(' '))))
    ##train[col]=train[col].apply(lambda x: " ".join(x) )
    #test[col]=test[col].apply(lambda x: list( set(x.split(' '))))
    ##test[col]=test[col].apply(lambda x: " ".join(x) )
    
    #    Most of the time, the first steps of an NLP project is to "tokenize" your documents, which main purpose is to normalize our texts. The three fundamental stages will usually include:
    #
        #break the descriptions into sentences and then break the sentences into tokens
        #remove punctuation and stop words
        #lowercase the tokens
        #herein, I will also only consider words that have length equal to or greater than 3 characters
    
    
    # apply the tokenizer into the item descriptipn column
    
    #train['tokens'] = train[col].map(tokenize)
    #test['tokens'] = test[col].map(tokenize)
    
    #b=train[['item_description','tokens','name_desc']]
    #a=train[[col]]
    
    #  
    words=['no free shipping','free shipping', 'free ship', '\[rm\]', 'missing']
    #replace brand
    words=a+words+words2replace
    #pattern = '|'.join(words)
    
    
    #replace used words!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
    #for word in words:
    #    train[col]=train[col].str.replace(word, '')
    #    test[col]=test[col].str.replace(word, '')

a=[]

###################################################
#bins=[0]
#group_names=[]
#for i in range (0,31):
#    bins.append(1+i*0.21) 
#    group_names.append(1+i)
#bins.append(1000) 
#group_names.append(max(group_names)+1) 
#
#train['price_bin'] = pd.cut(train['price'], bins, labels=group_names).astype(int)
#
#

#______________________________________
print("extra columns ", (time.clock()-t))

print("-----------extra columns start ", (time.clock()-start))



#------memo---------------------
gc.collect()
#start_mem=train.memory_usage().sum()+test.memory_usage().sum()##+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
#memory_reduce(train)
#memory_reduce(test)
##memory_reduce(train_sp)
##memory_reduce(test_sp)
#end_mem=train.memory_usage().sum()+test.memory_usage().sum()##+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
#print('memory change: ',(end_mem-start_mem)/start_mem)
#-------------------------------

#_____________________________________________________________________________
'''
/////////////////////////////////////////
           MEAN Encoding
/////////////////////////////////////////   
'''


#col='cat2_brand_qu'
col='full_cat_brand'
suf='_Median'
#fast code
price_av=np.percentile(train['price'], 50)
df = train.groupby(col)[['price']].median().astype(np.float32)
df.columns = [col+suf]
#train.drop(['brandMedian'],axis=1)

train = train.join(df, on=col, how='left')
print("Qual", rmsle(train[col+suf], train['price']))
test = test.join(df, on=col, how='left').fillna(price_av)
train_ex[col+suf]=train[col+suf]
test_ex[col+suf]=test[col+suf]
train.drop([col+suf], axis=1, inplace=True)
test.drop([col+suf], axis=1, inplace=True)
df=pd.DataFrame()
#mean
suf='_Mean'
#price_av=np.percentile(train['price'], 50)
price_av=train['price'].mean()
df = train.groupby(col)[['price']].mean().astype(np.float32)
df.columns = [col+suf]
#train.drop(['brandMedian'],axis=1)

train = train.join(df, on=col, how='left')
print("Qual", rmsle(train[col+suf], train['price']))
test = test.join(df, on=col, how='left').fillna(price_av)
train_ex[col+suf]=train[col+suf]
test_ex[col+suf]=test[col+suf]
train.drop([col+suf], axis=1, inplace=True)
test.drop([col+suf], axis=1, inplace=True)
df=pd.DataFrame()

def mean_encode_col(tr, te, col):
    suf="_mean_en"
    price_av=tr['price'].mean()
    df = tr.groupby(col)[['price']].mean().astype(np.float32)
    df.columns = ['col'+suf]
    #train.drop(['brandMedian'],axis=1)
    
    tr = tr.join(df, on=col, how='left')
    print("Qual mean ", col , rmsle(tr['col'+suf], tr['price']))
    te = te.join(df, on=col, how='left').fillna(price_av)
    
    tr.drop([col], inplace=True)
    te.drop([col], inplace=True)
    
    df=pd.DataFrame()
    return tr, te

#    if len(lb)>0:
#        for col in lb:
#            mean_encode_col(train, test, col)
#    else:
#        for col in extra_col:
#            mean_encode_col(train, test, col)
#    
#___________________________________________________________
#Mean Encode
#1
me_extra=0
train_me=pd.DataFrame(index=range(train.shape[0]))
test_me=pd.DataFrame(index=range(test.shape[0]))
if me_extra==1:
    extra_col=extra_col[:5]
#    extra_cat=extra_cat[:7]
    extra_cat=['cat_main', 'category_name']

    for col in extra_col:
        for col2 in extra_cat:
            #price_av=train['price'].mean()
            price_av=np.percentile(train['price'], 33)
    #        price_av=(train['price'].mean()+train['price'].median())/2
    #        price_av=(train['price'].mean()+np.percentile(train['price'], 33))/2
    #        price_av=(np.percentile(train['price'], 50)+np.percentile(train['price'], 30))/2
    #        
            #price_std=train['price'].std()
            #concat
            cols=col+'_'+col2
            train[cols]=train_ex[col].astype(str)+train[col2].astype(str)
            test[cols]=test_ex[col].astype(str)+test[col2].astype(str)
            
            #mean
            column_name=cols+'_mean_target'
    
    #        df=train.groupby(cols)[['price']].mean()
    #        df.columns = [column_name]
    #        
    #        train = train.join(df, on=cols, how='left')
    #        test = test.join(df, on=cols, how='left').fillna(price_av)
    
            df=train.groupby(cols)['price'].mean().astype(np.float32)
            train_me[column_name]=train[cols].map(df).fillna(price_av).astype(np.float32)
            test_me[column_name]=test[cols].map(df).fillna(price_av).astype(np.float32)
    #        
    #        print ( (train[column_name]-train[column_name+'1']).sum() )
            
    
            
            test = test.drop([cols], axis=1)
            train = train.drop([cols], axis=1)
            #del means
            df=pd.DataFrame()
            
            #val[col+'_'+col2+'mean_target']=val[col+'_'+col2].map(means)
            print("Qual "+column_name, rmsle(train_me[column_name], train['price']))
            
    
            #Deviation
    #        column_name=cols+'_std_target'
    #
    #        df=train.groupby(cols)['price'].std()
    #        df.columns = [column_name]
    #   
    ##        train = train.join(df, on=col, how='left')
    ##        test = test.join(df, on=col, how='left').fillna(price_std)
    ##        
    #        train[column_name]=train[cols].map(df).fillna(price_std)
    #        test[column_name]=test[cols].map(df).fillna(price_std)
    #    
    #        train_sp[column_name]=train[column_name]
    #        test_sp[column_name]=test[column_name]
    #        
    #        
    #        test = test.drop([cols], axis=1)
    #        train = train.drop([cols], axis=1)
    #        df=pd.DataFrame()
    #        del df
    #        print("Qual std "+column_name, rmsle(train_sp[column_name], train['price']))
            
            gc.collect()     
    
train_me, ri_me_tr, col_me=panda_spicy(train_me)
test_me, ri_me_te, col_me=panda_spicy(test_me)
#slow way
#df=pd.merge(test, df, on=col, how='inner')
#df=train.groupby(col)['price'].median().sort_values(ascending=False)
#ind_list=list(df.index)
#train['brandMedian']=train.apply(axis = 1, func = lambda x: df[x[col]] if x[col] in ind_list else 1)
#test['brandMedian']=test.apply(axis = 1, func = lambda x: df[x[col]] if x[col] in ind_list else 1)
#train['brandMedian2']=train.groupby(col)['price'].transform('median')

#test['brandMedian2']=None
#test.set_index(col).loc[test['brandMedian2'].isnull(), 'brandMedian2']=train[col].map(df)
#
#test.set_index(col).brandMedian2.fillna(train.set_index(col).brandMedian2).reset_index()
#
#
#train_sp['brandMedian2']=train.groupby(col)['price'].transform('median')
#test_sp['brandMedian2']=train.groupby(col)['price'].transform('median')


#------memo---------------------
gc.collect()
#start_mem=train.memory_usage().sum()+test.memory_usage().sum()##+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
#memory_reduce(train)
#memory_reduce(test)
#
#end_mem=train.memory_usage().sum()+test.memory_usage().sum()##+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
#print('memory change: ',(end_mem-start_mem)/start_mem)
#-------------------------------



print('median price time ',time.clock() - t)  
print("--------median columns ", (time.clock()-start))
'''Train_ex with extra columns'''

print("conv to spicy")
train_ex, ri_ex_tr, col_ex=panda_spicy(train_ex)
test_ex, ri_ex_te, col_ex=panda_spicy(test_ex)
'''-----------------------------------'''



t = time.clock()

#______________________________________________________
print("encoder")
enc=2
lb=[]
if enc==1:
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    #???????????? new column????
    suf="_le"
    for col in ['brand_name', '0_category_name_gen']: #, '0_cat_main_gen','0_cat_m_s1_gen']:
        train[col+suf] = le.fit_transform(train[col]).astype(int)
        dic = dict(zip(le.classes_, le.transform(le.classes_)))
        test[col+suf]=test[col].map(dic).fillna(dic["missing"]).astype(int)
        #test[col] = le.transform(test[col])
    #    test[col].fillna(value=9999, inplace=True)
        l.append([col+suf])
    
else:    
    
    col_lb=[]
    train_lb=pd.DataFrame()
    train_lb=sp.csr_matrix(train_lb)
    test_lb=sp.csr_matrix(train_lb)
    
    def Label_Binariz(train, test, col, train_ex, test_ex, col_lb):
        X_tr=[]
        X_te=[]
        from sklearn.preprocessing import LabelBinarizer
        lb = LabelBinarizer(sparse_output=True) #sparse CSR format
        X_tr = lb.fit_transform(train[col])
        print('l',X_tr.shape)
        X_te=lb.transform(test[col])
        
        X_tr=sp.csr_matrix(X_tr)
        X_te=sp.csr_matrix(X_te)
        
        col_lb.extend([s + '_'+ col for s in lb.classes_])
        train_ex=sp.hstack((train_ex, X_tr), format='csr')
        test_ex=sp.hstack((test_ex, X_te), format='csr')
        del lb
        return train_ex, test_ex, col_lb
    
    for col in ['brand_name', '0_category_name_gen', '0_cat_main_gen']:#,'0_cat_m_s1_gen']:
        train_lb, test_lb, col_lb=Label_Binariz(train, test, col, train_lb, test_lb, col_lb)
      

        
#        X_tr=pd.DataFrame(X_tr)
#        X_te=pd.DataFrame(X_te)
#        
#        X_tr.index=train.index
#        X_te.index=test.index
#        X_tr=X_tr.add_suffix('_'+col)
#        X_te=X_te.add_suffix('_'+col)
#        lb=X_tr.columns
#        train=pd.concat([train,X_tr], axis=1)
#        test=pd.concat([test,X_te], axis=1)
        
    
    X_tr=pd.DataFrame()
    X_te=pd.DataFrame()

gc.collect()
#start_mem=train.memory_usage().sum()+test.memory_usage().sum()##+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
#memory_reduce(train)
#memory_reduce(test)
##memory_reduce(train_sp)
##memory_reduce(test_sp)
#end_mem=train.memory_usage().sum()+test.memory_usage().sum()##+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
#print('memory change: ',(end_mem-start_mem)/start_mem)

#____________________________________________________________

                                                          



'''
#####DROP#################################################
'''
#train.drop('category_name', axis=1, inplace=True)
#test.drop('category_name', axis=1, inplace=True)

'''
//////////////////////////////////////////////////////////
'''

                                                             
#s=test[['name','brand_name',"category_name",'brandMedian','brandMedian2','item_condition_id','item_description',
#         'item_description_set','item_description_new','item_description_bnwt', 'item_description_pic']]
##.sort_values(['price'], ascending=False)
#s=s.loc[(s['brand_name']=='adidas') & (s["category_name"]=='men/shoes/athletic')]
#


    
#
#test['cat_main']
#It appears that few item names include their categories, but many items include their brands.

#has description



#cloud = WordCloud(width=1440, height=1080).generate(" ".join(train['item_description']
#.astype(str)))
#plt.figure(figsize=(20, 15))
#plt.imshow(cloud)
#plt.axis('off')

#train=train.to_sparse()
#test=test.to_sparse()

X_train=pd.DataFrame(index=range(train.shape[0]))
X_test=pd.DataFrame(index=range(test.shape[0]))


col_nlp=[]
if nlp_yes==1:
    print ("NLP start")
    t=time.clock()

    #from string import punctuation
    #from nltk.stem.porter import PorterStemmer
  
    
    # a 2 sec slower
    #t=time.clock()
    
    #print(time.clock()-t)
    
    #remove stopwords

    #stop_words = stopwords.words('english')
    # + list(punctuation)
    ###--------------------------------
    


    
    #remove special chars
    #clear_words
    
#    ps=PorterStemmer()
#    
#    test[col]=test[col].apply(lambda x: [ps.stem(item) for item in re.findall(r"[\w']+", x) ])
#    train[col]=train[col].apply(lambda x: [ps.stem(item) for item in re.findall(r"[\w']+", x) ])    


#    test[col]=test[col].apply(lambda x: [ps.stem(item) for item in re.findall(r"[\w']+", x) if ps.stem(item) not in stop_words])
#    train[col]=train[col].apply(lambda x: [ps.stem(item) for item in re.findall(r"[\w']+", x) if ps.stem(item) not in stop_words])

   
    #from nltk.stem.wordnet import WordNetLemmatizer 
    #lem = WordNetLemmatizer()
    
#    print('Size of ps', sys.getsizeof(ps))   
#    ps=PorterStemmer()
#    print('Size of ps', sys.getsizeof(ps))
#    del ps
    gc.collect()
    
    #del words
    #del stop_words
     
#    print('remove stopwords %.1f '%(time.clock()-t))
#
#    



#corpus
#t=time.clock()
#for col in ['name','item_description']:
#    corpus=[]
#    l=train.shape[0]
#    ps=PorterStemmer()
#    for i in range(0,l):
#        #clean html tags
#        #from BeautifulSoup import BeautifulSoup
#        # 
#        #soup = BeautifulSoup(html)
#        #all_text = ''.join(soup.findAll(text=True))
#     #   text=re.sub('[^a-zA-Z]', ' ', train[col][i]) #keep only letters
#        text=str(train[col][i])
#            #split
#        text=text.split()
#            #remove words from stopwords
#            #use stem to 'loved->love'
#            #remove irrelevant words
#            
#        #stemming - Stemming algorithms attempt to automatically remove suffixes (and in some cases prefixes) 
#        #in order to find the “root word” or stem
#    
#        text=[ps.stem(word) for word in text ]  #if not word in set(stopwords.words('english'))
#        #make a string
#        text=' '.join(text)
#        corpus.append(text)

#print('%.1f corpus'%(time.clock()-t))





    #bigrams = [list(zip(x,x[1:])) for x in train[col].values.tolist().split(" ")]
    #__________________________________________________________________
    #Create the BAg of words Model
    t1=time.clock()
    col="name_desc"
    maxfeat=1000
    ngramrange=(1,2)
    mindf=0.0001
    maxdf=0.5
    s=3
    #tokenizing


#    X_train.index=train.index
#    X_test.index=test.index
    for col in ["item_description", "name"]:
        gc.collect()
        if s==1 :
            from sklearn.feature_extraction.text import CountVectorizer
            X_te=pd.DataFrame()
            X_tr=pd.DataFrame()
            cv=CountVectorizer(tokenizer=tokenize,
                               max_features=maxfeat,  #will leave only 1500 words
                               ngram_range=ngramrange
                               , analyzer='word' 
                               , lowercase=False
                               , stop_words="english"
                               , min_df=mindf 
                               ,  max_df=maxdf
                               )
            X_tr= cv.fit_transform(train[col].astype(str))
            col_nlp.extend([s+"_"+col for s in cv.get_feature_names()])        
            X_te = cv.transform(test[col].astype(str))         
   
            
#            X_tr=pd.DataFrame(X_tr.toarray(), columns=cv.get_feature_names(),index=train_id)#.to_sparse()
#            X_tr = X_tr.add_suffix(col)
#            X_train=X_train.to_sparse()
            

#            X_te=pd.DataFrame(X_te.toarray(), columns=cv.get_feature_names(),index=test_id) #.to_sparse()
#            X_te = X_te.add_suffix(col)   
#            X_test=X_test.to_sparse()
            
            del cv
        elif s==2:
            from sklearn.feature_extraction.text import TfidfVectorizer
            X_test=pd.DataFrame()
            X_train=pd.DataFrame()
            tfidf=TfidfVectorizer(tokenizer=tokenize,
                                    lowercase=False
                                    , ngram_range=ngramrange
                                    ,analyzer='word' 
                                    , max_features=maxfeat
                                    , min_df=mindf
                                    , max_df=maxdf
                                    ,stop_words="english"
                                    )
    
           
            X_tr = tfidf.fit_transform(train[col].astype(str)).astype(np.float32)
            X_te = tfidf.transform(test[col].astype(str)).astype(np.float32)
 
            col_nlp.extend([s+"_"+col for s in tfidf.get_feature_names()])
 
#            X_tr=pd.DataFrame(X_tr.toarray(), columns=tfidf.get_feature_names(),index=train_id) #.to_sparse()
#            X_tr = X_tr.add_suffix(col)
#            X_te=pd.DataFrame(X_te.toarray(), columns=tfidf.get_feature_names(),index=test_id) #.to_sparse()
#            X_te = X_te.add_suffix(col) 

            del tfidf
        else:
            from sklearn.pipeline import Pipeline
            from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
            vect = CountVectorizer(tokenizer=tokenize,
                                       max_features=maxfeat,  #will leave only 1500 words
                                       ngram_range=ngramrange
                                       , analyzer='word' 
                                       , lowercase=False
                                       , stop_words="english"
                                       , min_df=mindf 
                                       , max_df=maxdf)
            tfidf = TfidfTransformer(norm='l2', use_idf=True, sublinear_tf=True)
            pipeline = Pipeline([
                ('vect',vect),
                ('tfidf',tfidf)])
            X_tr = pipeline.fit_transform(train[col].astype(str)).astype(np.float32)
            X_te = pipeline.transform(test[col].astype(str)).astype(np.float32)
            col_nlp.extend([s+"_"+col for s in vect.get_feature_names()])
            
            del tfidf, vect, pipeline
        print('nlp time column ', time.clock()-t)
#        X_tr.index=train.index
#        X_train=pd.concat([X_train,X_tr], axis=1)
#        X_te.index=test.index
#        X_test=pd.concat([X_test,X_te], axis=1)   
        X_train=sp.hstack((X_train,X_tr))
        X_test=sp.hstack((X_test,X_te))
        
#            train.drop([col], axis=1, inplace=True)
#            test.drop([col], axis=1, inplace=True)
        gc.collect()
        
    print("cv ",X_train.shape)

#    memory_reduce(X_train)
#    memory_reduce(X_test)
    X_tr=pd.DataFrame()    
    X_te=pd.DataFrame()
    
    gc.collect()
    

    
    
#    df=pd.DataFrame(columns=['train_id','category_name','item_description','name','brand_name', 'shipping', 'item_condition_id','price'],index=train_id)
#    for col in df.columns:
#        df[col]=train[col]
        

    t1=time.clock() - t1
    print('NLP time ',time.clock() - t1)  


#    l=0
#    for col in X_train.columns:
#        l+=1
#        cr=train['price'].corr(X_train[col])
#        if abs(cr)<0.05 or np.isnan(cr):
#    #        print('drop')
#            X_train.drop([col], axis=1)  
#    #        X_test.drop([col], axis=1)
#        else:
#            print(l," correlation with token ", col, " is ",  cr)

'''
-------------------------------------------------------------------------------------
'''
prepr_time=time.clock()-start
print("preprocess time: ", prepr_time)
#Training and TEst data ---------------------------------
print("Prepare data")
y_train=np.array(train['price'])
train.drop("price", axis=1, inplace=True)

#y_train_bin=np.array(train['price_bin'])
#train.drop("price_bin", axis=1, inplace=True)

test.drop("test_id", axis=1, inplace=True)
train.drop("train_id", axis=1, inplace=True)


print(X_train.shape , train.shape)
#train.to_dense()
#test.to_dense()


'''---FINAL MATRIX Sparse-------------------------'''
X_train=sp.hstack((train_ex, train_lb, train_dum, train_me, X_train)).tocsr() 
X_test=sp.hstack((test_ex, test_lb, test_dum, test_me, X_test)).tocsr() 

print ("X_train :" , X_train.shape)
#columns
print("ex ", len(col_ex),"LB ", len(col_lb),"dum ", len(col_dum),
      "mean ", len(col_me), "nlp ", len(col_nlp), 'sum ', len(col_ex)+len(col_lb)+len(col_dum)+len(col_me)+len(col_nlp))
train_me.shape
#integers
cols=[]
cols.extend(col_ex)
col_ex=[]
cols.extend(col_lb)
col_lb=[]
cols.extend(col_dum)
col_dum=[]

cols_i=cols


#float32
cols.extend(col_me)
col_me=[]
cols.extend(col_nlp)
col_nlp=[]

M=len(cols)
#index

topan=0
if topan==1:
    '''---FINAL MATRIX dense-------------------------'''
    print('Convert sparse')
    t=time.clock()
    X_train=sparse_pandas(X_train, ri_ex_tr, cols).to_sparse(fill_value=0)
    X_test=sparse_pandas(X_test, ri_ex_te, cols).to_sparse(fill_value=0)
    print('Convert sparse to pandas time:', time.clock()-t)
    

#t=time.clock()
#def col_to_int8(df, col_list):
#    for c in col_list:
#        df[c] = df[c].astype(np.int8)
#        gc.collect()
#    return df
#
#col_to_int8(X_train, cols_i)
#col_to_int8(X_test, cols_i)    
#print('Convert sparse to int8 time:', time.clock()-t) 
#cols=[]
#
#start_mem=X_train.memory_usage().sum()+X_test.memory_usage().sum()##+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
#memory_reduce(X_train)
#memory_reduce(X_test)
##memory_reduce(train_sp)
##memory_reduce(test_sp)
#end_mem=X_train.memory_usage().sum()+X_test.memory_usage().sum()##+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
#print('memory change: ',(end_mem-start_mem)/start_mem)
gc.collect()

'''-------------------------'''

'''---------Ridge --------------------------'''
if ridg==1:
    t=time.clock() 
    from sklearn.linear_model import Ridge #, LogisticRegression
    
    
    print("Fitting Model")
    #model.fit(X_train, y_train)
    #model.coef_
    #y_pred_tr = model.predict(X_train)
    
    #s=int(len(y_train)/2)
    #regr_ri = Ridge( alpha=0.005,  tol=0.001,
    #              max_iter=1000, normalize=True,
    #              fit_intercept=True) #solver = "lsqr",
    #
    #y_pred_ri,y_pred_ri_test =model_predict(regr_ri,X_train, y_train, X_test)
    #print("*Error ridge 1", rmsle(np.expm1(y_pred_ri) ,np.expm1(y_train)))
    
    gc.collect()
    s=int(len(y_train)/2)
    regr_ri = Ridge( alpha=0.005,  tol=0.001,
                  max_iter=500, normalize=True,
                  fit_intercept=True) #solver = "lsqr",
    
    y_pred_ri,y_pred_ri_test =model_predict(regr_ri,X_train[:s], y_train[:s], X_test)
    print("*Error ridge 1", rmsle(np.expm1(y_pred_ri*std+mean) ,np.expm1(y_train[:s]*std+mean)))
    y_pred_ri=[]
    print("ridge 1 time:", (time.clock()-t))
    t=time.clock() 
    regr_ri = Ridge( alpha=0.05,  tol=0.001,
                  max_iter=500, normalize=True,
                  fit_intercept=True) #solver = "lsqr",
    y_pred_ri,y_pred_ri_test2 =model_predict(regr_ri,X_train[s:], y_train[s:], X_test)
    print("*Error ridge 2", rmsle(np.expm1(y_pred_ri*std+mean) ,np.expm1(y_train[s:]*std+mean)))
    print("ridge 2 time:", (time.clock()-t))
    
    y_pred_ri=[]
    y_pred_ri_test=(y_pred_ri_test+y_pred_ri_test2)/2.0
    y_pred_ri_test2=[]

#    from sklearn.utils import shuffle
#    from sklearn.model_selection import KFold
#    def cross_validate(model, x, y, folds=10, repeats=5):
#        '''
#        Function to do the cross validation - using stacked Out of Bag method instead of averaging across folds.
#        model = algorithm to validate. Must be scikit learn or scikit-learn like API (Example xgboost XGBRegressor)
#        x = training data, numpy array
#        y = training labels, numpy array
#        folds = K, the number of folds to divide the data into
#        repeats = Number of times to repeat validation process for more confidence
#        '''
#        ypred = np.zeros((len(y),repeats))
#        score = np.zeros(repeats)
#        x = np.array(x)
#        for r in range(repeats):
#            i=0
#            print('Cross Validating - Run', str(r + 1), 'out of', str(repeats))
#            x,y = shuffle(x,y,random_state=r) #shuffle data before each repeat
#            kf = KFold(n_splits=folds,random_state=i+1000) #random split, different each time
#            for train_ind,test_ind in kf.split(x):
#                print('Fold', i+1, 'out of',folds)
#                xtrain,ytrain = x[train_ind,:],y[train_ind]
#                xtest,ytest = x[test_ind,:],y[test_ind]
#                model.fit(xtrain, ytrain)
#                ypred[test_ind,r]=model.predict(xtest)
#                i+=1
#            score[r] =rmsle(np.expm1(ypred[:,r]) ,np.expm1(y)) # R2(ypred[:,r],y)
#        print('\nOverall score:',str(score))
#        print('Mean:',str(np.mean(score)))
#        print('Deviation:',str(np.std(score)))
#        pass
#    
#    cross_validate(regr_ri, np.array(X_train), y_train, folds=5, repeats=2)
#    pass







#train=train._get_numeric_data()
#test=test._get_numeric_data()
#
#
##X_train.to_dense()
##X_test.to_dense()
#sys.getsizeof(X_train)
#
##X_train=X_train.rename(index=str, columns={"fit": "fit_re"})
##X_test=X_test.rename(index=str, columns={"fit": "fit_re"})
##
##X_train=X_train.rename(index=str, columns={"shipping": "ship_re"})
##X_test=X_test.rename(index=str, columns={"shipping": "ship_re"})
#
#print('test ',test.shape , 'train', train.shape, 'y',  y_train.shape)
#gc.collect()
#
#X_train.index=train.index
#X_test.index=test.index
#X_train=pd.concat([train, X_train], axis=1)
#X_test=pd.concat([test, X_test], axis=1)
#
#print('test ',X_test.shape , 'train', X_train.shape, 'y', y_train.shape)

gc.collect()
test=pd.DataFrame()
#train=pd.DataFrame()
gc.collect()
#del test
#del train



#c=set(X_test.columns)
#[x for x in X_train.columns if x not in c]




#test_id=X_test["test_id"]
#test_id=X_test.index

#cols=['name_desc',"name","brand_name", 'item_description','category_name',
#      'cat_main', 'cat_sub1', 'cat_sub2']
#for col in cols:
#    X_test=X_test.drop(col, axis=1)
#    X_train=X_train.drop(col, axis=1)
#    gc.collect()
    
#train_id=X_train["train_id"]
#train_id=X_train.index





#FREE YOUR MEMO--------------------------------------------------------------
t=time.clock()

#------memo---------------------
gc.collect()


#--------------------------------------------------------
#sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()
#print('[{}] Finished to create sparse merge'.format(time.time() - start_time))
#
#X = sparse_merge[:nrow_train]
#X_test = sparse_merge[nrow_train:]





word=[]
words=[]
#del a
#del b
print('memory time ',time.clock() - t)  
#### Model---------------------------------------------------


#------------------------------------------------------------


#import xgboost as xgb
#regr_xgb=xgb.XGBRegressor()



#_____________________________________
if lgbm==1:
    t=time.clock()
    
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split #, cross_val_score
    
    gc.collect()
    
    train_X, valid_X, train_y, valid_y = train_test_split(X_train, 
                                                          y_train, 
                                                          test_size = 0.1, 
                                                          random_state = 121) 
    d_train = lgb.Dataset(train_X, label=train_y,max_bin= 8192)
    d_valid = lgb.Dataset(valid_X, label=valid_y,max_bin= 8192)
    watchlist = [d_train, d_valid]
    
    params = {
    	'learning_rate': 0.78,
    	'application': 'regression',
    	'max_depth': 3,
    	'num_leaves': 99,
    	'verbosity': -1,
    	'metric': 'RMSE',
    	'nthread': 4,
        
    }
    
    
    
    
    model = lgb.train(params, train_set=d_train, num_boost_round=7500, 
                      valid_sets=watchlist, 
                      early_stopping_rounds=50, 
                      verbose_eval=500) 
    
    
    print("Features importance...")
    gain = model.feature_importance('gain')
    ft = pd.DataFrame({'feature':model.feature_name(), 'split':model.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    #print(ft.head(100))
    
    
    predsL = model.predict(X_test)
    
    #    Use large max_bin (may be slower)
    #    Use small learning_rate with large num_iterations
    #    Use large num_leaves (may cause over-fitting)
    #    Use bigger training data
    #    Try dart
    del model
    print("lgbm 1 time:", (t-time.clock()))
    t=time.clock() 
    params = {
    	'learning_rate': 0.1,
    	'application': 'regression',
    	'max_depth': 7,
    	'num_leaves': 99,
    	'verbosity': -1,
    	'metric': 'RMSE',
    	'nthread': 4
    }
    
    model = lgb.train(params, train_set=d_train, num_boost_round=10000, 
                      valid_sets=watchlist, 
                      early_stopping_rounds=1000, verbose_eval=500) 
    
    print("Features importance...")
    gain = model.feature_importance('gain')
    ft2 = pd.DataFrame({'feature':model.feature_name(), 'split':model.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    #print(ft.head(100))
    
    predsL2_train = model.predict(X_train).astype(np.float32)
    
     
    
    #a=pd.DataFrame(columns=['pred','dif'], index=train_id)
    #a['pred']=predsL2_train
    #a['dif']=y_train-predsL2_train
    #a.index=df.index
    
    #df.drop(['dif','pred'], axis=1, inplace=True)
    #df=pd.concat([df, a], axis=1)
    #df=df.sort_values(['dif'], ascending=False)
    
    predsL2 = model.predict(X_test).astype(np.float32)
    
    del model
    print('[{}] Finished to predict lgb 2'.format(time.clock() - t))



#------memo---------------------
gc.collect()
#-------------------------------
#ANN -NEURO NETS
if ann==1:
    print('start ann')
    t=time.clock()
    from keras import backend as K
    import keras
    import tensorflow as tf
    #print (tensorflow.__version__)
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.wrappers.scikit_learn import KerasRegressor
    
    from keras.optimizers import RMSprop
    rmsprop = RMSprop(lr =0.0001)
    
    from sklearn.preprocessing import MinMaxScaler
    sc= MinMaxScaler()
    y_train=sc.fit_transform(y_train.reshape(-1,1))
    #X_train.reset_index(drop=True, inplace=True)
    
    def rmsle_K(y, y0):
        return K.sqrt(K.mean(K.square(tf.log1p(y) - tf.log1p(y0))))

    #X_train=np.array(X_train)
    '''!!!!!!!!!!!!!!!!!!!!!!'''
    #M=X_train.shape[1]
    N=int(M/2)
    N2=int(N/2)
    N2=256
    #initialize ANN
    model=Sequential()
    #Step1 - Dense will make weights
    #Spet2 -Input nodes = indepentent variables =11
    #first hidden layer
    inp=M
    model.add(Dense(N, activation="relu", 
                         kernel_initializer="uniform", #initialize weights
                         #units=N,        #number of node in hidden layer (mozhno ~(1+11)/2)
                         #parameter tuning part10
                         input_dim=inp))
    
    #model.add(Dropout({{uniform(0, 1)}}))
    
    #secondhidden layer
    inp=N
    model.add(Dense(N2, input_dim=inp,
                        activation="relu", 
                        kernel_initializer="uniform" #initialize weights
                                #number of node in hidden layer (mozhno ~(1+11)/2)
                         #parameter tuning part10
                         ))
    
    #model.add(Dropout(rate=0.15))
#    #step3 - Activation function - sigmoid to output layer and 
#    #Threshold function for other layers (Rectifier)
    #Final layer
    inp=N2
    model.add(Dense(1, input_dim=inp, activation="sigmoid", 
                         kernel_initializer="uniform", #initialize weights
                         #parameter tuning part10
                         ))
    #if at output layer you have more than 2 categories - "softmax" activation
    #classifiar.add(Dense(units = 6 , 
    #                     use_bias=True, 
    #                     kernel_initializer='glorot_uniform', 
    #                     bias_initializer='zeros', 
    #                     activation='relu' , input_dim =11) )
    
    
    #Step4 -Compare to real data
    #compile
    model.compile(optimizer=rmsprop,# algoritm to calculate weights (Stochastic Gradient Descend)
                       loss='mean_squared_logarithmic_error', #within optimizer algorithm (logarithmic loss) catogorical_crossentropy
#                           metrics=['mean_squared_logarithmic_error']
                       metrics=[rmsle_K]
                       )
    
    
    #create layers
    def baseline_model():
        model=Sequential()
        #Step1 - Dense will make weights
        #Spet2 -Input nodes = indepentent variables =11
        #first hidden layer
        inp=M
        model.add(Dense(N, activation="relu", 
                             kernel_initializer="uniform", #initialize weights
                             #units=N,        #number of node in hidden layer (mozhno ~(1+11)/2)
                             #parameter tuning part10
                             input_dim=inp))
        
        #model.add(Dropout({{uniform(0, 1)}}))
        
        #secondhidden layer
        inp=N
        model.add(Dense(N2, input_dim=inp,
                            activation="relu", 
                            kernel_initializer="uniform" #initialize weights
                                    #number of node in hidden layer (mozhno ~(1+11)/2)
                             #parameter tuning part10
                             ))
    #    #step3 - Activation function - sigmoid to output layer and 
    #    #Threshold function for other layers (Rectifier)
        #Final layer
        
                #secondhidden layer
        inp=N2
        model.add(Dense(N2, input_dim=inp,
                            activation="relu", 
                            kernel_initializer="uniform" #initialize weights
                                    #number of node in hidden layer (mozhno ~(1+11)/2)
                             #parameter tuning part10
                             ))
    #    #step3 - Activation function - sigmoid to output layer and 
    #    #Threshold function for other layers (Rectifier)
        #Final layer
        
        inp=N2
        model.add(Dense(1, input_dim=inp, activation="sigmoid", 
                             kernel_initializer="uniform", #initialize weights
                             #parameter tuning part10
                             ))
        #if at output layer you have more than 2 categories - "softmax" activation
        #classifiar.add(Dense(units = 6 , 
        #                     use_bias=True, 
        #                     kernel_initializer='glorot_uniform', 
        #                     bias_initializer='zeros', 
        #                     activation='relu' , input_dim =11) )
        
        
        #Step4 -Compare to real data
        #compile
        model.compile(optimizer='rmsprop',# algoritm to calculate weights (Stochastic Gradient Descend)
                           loss='mean_squared_logarithmic_error', #within optimizer algorithm (logarithmic loss) catogorical_crossentropy
#                           metrics=['mean_squared_logarithmic_error']
                           metrics=[rmsle_K]
                           )
        
        return model
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # evaluate model with standardized dataset
#    regr_ann = KerasRegressor(build_fn=baseline_model, 
#                              epochs=3, 
#                              batch_size=64
#                              #, validation_data=(x_test, y_test)
#                              
#                              )
    
    def batch_generator(X, y, batch_size, shuffle):
        number_of_batches = np.ceil(X.shape[0]/batch_size)
        counter = 0
        sample_index = np.arange(X.shape[0])
        
        if shuffle:
            np.random.shuffle(sample_index)
        while True:
            batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
            X_batch = X[batch_index,:]#.toarray()
            y_batch = y[batch_index]
            counter += 1
            yield X_batch, y_batch
            if (counter == number_of_batches):
                if shuffle:
                    np.random.shuffle(sample_index)
                counter = 0
    
    #fit to trainig set
    #train on all data
    #regr_ann.fit(x=X_train,y=y_train)
    #train on batches
    batch_size=64
    #fit first batch
    ba=64
    nb_epoch=3
    X_train.shape[1]
#    model.fit(X_train[:ba], y_train[:ba], epochs=nb_epoch, batch_size=batch_size, verbose=1)
    model.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size, verbose=1
              ,validation_split=0.1)
#    s=X_train.shape[0]
#    model.trainable = True
#    for e in range(nb_epoch):
#        n=1
#        while (n+1)*ba<s:
#            
#    #        x_ba=pd.DataFrame(x_ba.todense())
#            logs =model.train_on_batch(X_train[n*ba:(n+1)*ba].toarray(), y_train[n*ba:(n+1)*ba])
#            n+=1
#            if (n%100)==0:
#                print('n ',n , 'errors', logs)
#
#        #last pies 
#        if (s % ba)>0:
#            logs =model.train_on_batch(X_train[n*ba:s].toarray(), y_train[n*ba:s])
#            print('n ',n , 'errors', logs)
         
#    model.fit_generator(generator=batch_generator(X_train, y_train, batch_size, False),
#                           samples_per_epoch = len(X_train)
#                           #,use_multiprocessing=True
#                           
#                           ,epochs =2
#                           #, verbose=1
##                           validation_data=batch_generator(X_test, Y_test, batch_size, False),
##                           nb_val_samples=len(X_test), 
#                          # max_q_size=20
#                           , nb_worker=4
#                           )
    
    
    
    #Predict
    if hm==1:
        y_pred_ann_train=model.predict(X_train)
        y_pred_ann_train=sc.inverse_transform(y_pred_ann_train)
        y_pred_ann_train=np.hstack(y_pred_ann_train)
        
        y_train=sc.inverse_transform(y_train)
        y_train=np.hstack(y_train)
        
        print("*Error ANN ", rmsle(np.expm1(y_pred_ann_train*std+mean) ,np.expm1(y_train*std+mean)))
        
    y_pred_ann=model.predict(X_test) 
    y_pred_ann=sc.inverse_transform(y_pred_ann)
    y_pred_ann=np.hstack(y_pred_ann)
    del model
    print('[{}] Finished to predict ANN'.format(time.clock() - t))

#XGB
#y_pred_xgb,y_pred_xgb_test =model_predict(regr_xgb,X_train, y_train, X_test)
#
#print("*Error XGB ", rmsle(np.round(np.expm1(y_pred_xgb)), np.expm1(y_train)))
#y_pred=(2*y_pred_xgb+y_pred_ri)/3


#print("*Error combo ", rmsle(np.round(np.expm1(y_pred)), np.expm1(y_train)))
#predict test

#y_pred_test=(2*y_pred_xgb_test+y_pred_ri_test)/3
    
X_test=pd.DataFrame()
    
if ridg==1:
    y_pred_test=y_pred_ri_test
    #y_pred_test=(y_pred_xgb_test+y_pred_ri_test +predsL)/3
    #y_pred_test = np.exp(y_pred_test)-1
    y_pred_test=y_pred_test*std+mean
    y_pred_test=np.expm1(y_pred_test).astype(np.float32)
    
    #y_pred_test=np.round(y_pred_test)
    
    #make submission
    X_test["test_id"]=test_id.astype(int)
    #1
    X_test['price']=y_pred_test
    #X_test['price2']=X_test['price'].round().astype(int)
    
    X_test[["test_id", "price"]].to_dense().to_csv("submission.csv", index = False, sep=',', encoding='utf-8')
    #X_test[["test_id", "price"]].toCSV("submission.csv")

    #2
    X_test['price']=np.round(y_pred_test)
    #X_test['price2']=X_test['price'].round().astype(int)
    X_test[["test_id", "price"]].to_dense().to_csv("submission_round.csv", index = False, sep=',', encoding='utf-8')
    


#3 LGBM
if lgbm==1:  
    X_test["test_id"]=test_id.astype(int)
    
    y_pred_test=(predsL+predsL2)/2
    y_pred_test=y_pred_test*std+mean
    predsL=np.expm1(y_pred_test).astype(np.float32)
    
    X_test['price']=predsL
    #X_test.drop(['price'], axis=1)
    #X_test.rename(columns={'priceLGBM': 'price'}, inplace=True)
    X_test[["test_id", "price"]].to_dense().to_csv("submissionLGBM.csv", index = False, sep=',', encoding='utf-8')
    
    #4
    X_test['price']=np.round(predsL)
    #X_test.drop(['price'], axis=1)
    #X_test.rename(columns={'priceLGBM': 'price'}, inplace=True)
    X_test[["test_id", "price"]].to_dense().to_csv("submissionLGBMround.csv", index = False, sep=',', encoding='utf-8')


#4 ANN  ----------------------------------------------------------------------------
if ann==1:
    X_test["test_id"]=test_id.astype(int)
    y_pred_test=y_pred_ann
    y_pred_test=y_pred_test*std+mean
    y_pred_test=np.expm1(y_pred_test).astype(np.float32)
    
    X_test['price']=y_pred_test
    X_test.loc[X_test['price'] <3, 'price'] = 3
    
    
    #X_test.drop(['price'], axis=1)
    #X_test.rename(columns={'priceLGBM': 'price'}, inplace=True)
    X_test[["test_id", "price"]].to_dense().to_csv("submissionANN.csv", index = False, sep=',', encoding='utf-8')
    
    #4
    X_test['price']=np.round(y_pred_test)
    #X_test.drop(['price'], axis=1)
    #X_test.rename(columns={'priceLGBM': 'price'}, inplace=True)
    X_test[["test_id", "price"]].to_dense().to_csv("submissionANNround.csv", index = False, sep=',', encoding='utf-8')

time.clock() - start
print('----training time ',time.clock() - t) 
full_time=time.clock() - start

print('All done in ',full_time, "preprocess took :" , prepr_time/(full_time))  

#print('NLP ',t1/full_time, "dum :" , t2/(full_time))

#if __name__ == '__main__':
#    main()

#X_train.columns.T