#-*- coding: utf-8 -*-
"""
Created on Thu Nov 30 15:51:51 2017

@author: Yury
"""

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
gc.collect()


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
    df[col+"_"+word]=0
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
    return regr.predict(xtrain), regr.predict(xtest) 

def map_column(df, dic, col, suf):
    df[col+suf] = df[col].map(dic).fillna("missing")
    return df

def main():


    #------------------------------------------------
        
    #train =pd.read_csv("train.tsv", sep="\t",  engine="python",encoding='utf-8')
    #test = pd.read_csv("test.tsv", sep="\t",  engine="python",encoding='utf-8')
    #train=train[:100000]
    #test=test[:100000]
    ##
    #test.to_csv("test_short.csv", index = False)
    #train.to_csv("train_short.csv", index = False)
    
#    train =pd.read_csv("train_short.csv",  engine="python")
#    test = pd.read_csv("test_short.csv",  engine="python")
    
    train =pd.read_csv("../input/train.tsv", sep="\t",  engine="python")
    test = pd.read_csv("../input/test.tsv", sep="\t",  engine="python")
    start_mem_tr=train.memory_usage().sum()
    #_____________________________________
    add_dummies=0
    nlp_yes=1
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
    a.plot()    
    
    print(train['price'].groupby(train['item_condition_id']).quantile(.40))
    print(train['price'].groupby(train['item_condition_id']).std())
    
    dic={1:1,2:1,3:1, 4:2,5:3}
    map_column(train, dic, 'item_condition_id',"_gen")
    map_column(test, dic, 'item_condition_id',"_gen")
    
    a=train[['name', 'brand_name','item_description','price']].sort_values(['price'], ascending=False)
    train=train.loc[train[col]>0] 
    av_price=np.percentile(train['price'], 99)
    train=train[train[col]<av_price]  
    
    train.reset_index(drop=True, inplace=True)
    train[col] = np.log1p(train[col])
    
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
    extra_cat.append(col+ "_gen")  
    
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
    extra_cat.append(col+ "_gen")   
    
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
        t=time.clock()
        l=0
        
        for col in ["brand_name",'item_condition_id']:#,'cat_main','cat_sub1','cat_sub2']:
            l+=1
            X_train = pd.get_dummies(train[col].astype(str)).astype(np.int8).to_sparse()
            print("dum ",X_train.shape)
            X_test = pd.get_dummies(test[col].astype(str)).astype(np.int8).to_sparse()
            #re-index the new data to the columns of the training data, filling the missing values with 0
            X_test=X_test.reindex(columns = X_train.columns, fill_value=0)
            
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
            
            X_train=pd.DataFrame(X_train).add_suffix("_"+str(l))
            X_test=pd.DataFrame(X_test).add_suffix("_"+str(l))
    
            train=pd.concat([train, X_train], axis=1)
    
            test=pd.concat([test,X_test], axis=1)
            print("dum2",train.shape)
    #    del train_X
    #    del test_X
        #print(train_X.info())
        print('Size of dataframe', sys.getsizeof(X_train))
        X_train=pd.DataFrame()
        #memory_reduce(train_X)
        #print(train_X.info())
        print('Size of dataframe', sys.getsizeof(X_train))
        X_test=pd.DataFrame()
        print("dummies ", (time.clock()-t))
    
    
    #------memo---------------------
    gc.collect()
    start_mem=train.memory_usage().sum()+test.memory_usage().sum() ###+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
    memory_reduce(train)
    memory_reduce(test)
    #memory_reduce(train_sp)
    #memory_reduce(test_sp)
    end_mem=train.memory_usage().sum()+test.memory_usage().sum() ###+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
    print('memory change: ',(end_mem-start_mem)/start_mem)
    
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
    col='full_cat_brand'
    train[col]=train["category_name"].astype(str)+train['brand_name'].astype(str)+train['item_condition_id'].astype(str)
    test[col]=test["category_name"].astype(str)+test['brand_name'].astype(str)+test['item_condition_id'].astype(str)
    extra_cat.append(col)
    
    #"category_name"+brand+qu
    col='full_cat_brand_ship'
    train[col]=train["category_name"].astype(str)+train['brand_name'].astype(str)+train['shipping'].astype(str)
    test[col]=test["category_name"].astype(str)+test['brand_name'].astype(str)+test['shipping'].astype(str)
    extra_cat.append(col)
    
    print('cat_brand combinations %.3f seconds ' %(time.clock() -t) ) 
    
    
    
    
    #------memo---------------------
    gc.collect()
    start_mem=train.memory_usage().sum()+test.memory_usage().sum()##+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
    memory_reduce(train)
    memory_reduce(test)
    
    end_mem=train.memory_usage().sum()+test.memory_usage().sum()##+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
    print('memory change: ',(end_mem-start_mem)/start_mem)
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
    suf='_yes'
    for col in ["brand_name",'item_description']:
        extra_col.append(col+suf)
        test[col+suf]=1
        test.loc[test[col].str.count('missing')>0, col+suf] =0
        
    #    test_sp[col+suf]=(train[col]!='missing')*1
        
        train[col+suf]=1
        train.loc[train[col].str.count('missing')>0, col+suf] =0
        
        
        
    #train['HasDescription']=(train['item_description']!='missing')*1
    #test['HasDescription']=(test['item_description']!='missing')*1
      
        
    #has brand in descr
    col="brand_name"
    col2='item_description'
    test['hasBrandDesc']=(test.apply(axis = 1, func = lambda x: str(x[col]).lower() in str(x[col2])))*1
    train['hasBrandDesc']=(train.apply(axis = 1, func = lambda x: str(x[col]).lower() in str(x[col2])))*1
    
    extra_col.append('hasBrandDesc')
    
    
    #most expensive brands
    
    
    #most expensive categories
    
    
    
    words2replace=[]
    #with photo
    # Add whether it may have pictures
    #pic_word_re = [re.compile(r, re.IGNORECASE) for r in [r'(see(n)?)?( in| the| my) (picture(s)?|photo(s)?)']]
    words=['photos','pictures', ' pics ', 'pix', 'image', 
           'photograph', 'can see', 'can look', 'can view', 'may see', 
           'may look', 'may view']
    words2replace+=words
    pattern = '|'.join(words)
    suf='_pic'
    for col in ["name",'item_description']:
        extra_col.append(col+suf)
        test[col+suf]=0
        test.loc[test[col].str.count(pattern)>0, col+suf] =1
        
        train[col+suf]=0
        train.loc[train[col].str.count(pattern)>0, col+suf] =1
        
        
    #several items
    words=['pairs', 'bundle', 'set', 'pcs', 'peaces', 'kit', 'batch', 
           'collection', 'pack', 'band', 'group', 'items', 'array',
           'assortment', 'each', 'includes', 'comes with', 'all in']
    words2replace+=words
    pattern = '|'.join(words)
    suf='_set'
    for col in ["name",'item_description']:
        extra_col.append(col+suf)
        test[col+suf]=0
        test.loc[test[col].str.count(pattern)>0, col+suf] =1
        
        train[col+suf]=0
        train.loc[train[col].str.count(pattern)>0, col+suf] =1
        
    
    #contains [rm]
        
    suf='_rm'
    for col in ["name",'item_description']:
        extra_col.append(col+suf)
        test[col+suf]=0
        test.loc[test[col].str.count('\[rm\]')>0, col+suf] =1
        
        
        train[col+suf]=0
        train.loc[train[col].str.count('\[rm\]')>0, col+suf] =1
        
    
    #brand new without tags
    words=["bnib", "bnwt", 'b.n.w.t.',
          "bnwot", 'b.n.w.o.t.', "refurbished", 
          'like new', 'worn', 'times', 'condition', 'cleaning', 'once', 'twice']
    words2replace+=words
    pattern = '|'.join(words)
    suf='_bnwt'
    for col in ["name",'item_description']:
        extra_col.append(col+suf)
        test[col+suf]=0
        test.loc[test[col].str.count(pattern)>0, col+suf] =1
        
        train[col+suf]=0
        train.loc[train[col].str.count(pattern)>0, col+suf] =1
        
    
    #new
    words=['nwt', "N.W.T.", 'n.w.t.', "new with tag", 'unused', 'never used', 'never opened', 
         'new in box', 'unopened']
    words2replace+=words
    pattern = '|'.join(words)
    suf='_new'
    for col in ["name",'item_description']:
        extra_col.append(col+suf)
        test[col+suf]=0
        test.loc[test[col].str.count(pattern)>0, col+suf] =1
        
        train[col+suf]=0
        train.loc[train[col].str.count(pattern)>0, col+suf] =1
        
    
    #china
    words=['made in china', 'cheap', 'low', 'bad']
    words2replace+=words
    
    
    #authentic
    words=['authentic', '100%','excellent', 'genuine', 'vintage', 'perfect', 'unique', 'antique set', 'solid gold', 'edition', 'limited edition', 'original']
    words2replace+=words
    pattern = '|'.join(words)
    suf='_auth'
    for col in ["name",'item_description']:
        extra_col.append(col+suf)
        test[col+suf]=0
        test.loc[test[col].str.count(pattern)>0, col+suf] =1
        
        train[col+suf]=0
        train.loc[train[col].str.count(pattern)>0, col+suf] =1
    
    words=['authentic',
    'bnwt',
    'box',
    'comes',
    'condition',
    'free',
    'gorgeous',
    'hardware',
    'htf',
    'iphone',
    'new',
    'nwt',
    'original',
    'paid',
    'price',
    'rare',
    'retail',
    'save',
    'shipping',
    'signature',
    'sticker',
    'tags',
    'tracking',
    'worn',
    'zip']
    
    for word in words:
        for col in ["name",'item_description']:
            extra_col.append(col+'_'+word)
            test[col+'_'+word]=0
            test.loc[test[col].str.contains(word), col+'_'+word] =1
            
            train[col+'_'+word]=0
            train.loc[train[col].str.contains(word), col+'_'+word] =1
    
    
    
    
    #for col in ["name",'item_description']:
    #    for word in words:
    #        rmsle_study(train, col, word, 0.05)
    
    
    #for col in ["name",'item_description']:
    #    for word in words2replace:
    #        corr_study(train, col, word, 0.05)   
    
    #plt.scatter(train['price'],train['item_description_auth'])
    #print("correlation ",  train['price'].corr(train['item_description_N.W.T.']))
    #
    #______________________________________
    #join and clear name and descr
    col='name_desc'
    test[col]=test['name'].astype(str)+(" ")+test['item_description'].astype(str)
    train[col]=train['name'].astype(str)+(" ")+train['item_description'].astype(str)
    
    #replace symbols
    train[col] = train[col].map(lambda x: re.sub('[^a-zA-Z ]', '', x))
    test[col] = test[col].map(lambda x: re.sub('[^a-zA-Z ]', '', x))
    
    
    #Tokenize???? split(' ') and clear duplicates
    
    train[col]=train[col].apply(lambda x: list( set(x.split(' '))))
    train[col]=train[col].apply(lambda x: " ".join(x) )
    test[col]=test[col].apply(lambda x: list( set(x.split(' '))))
    test[col]=test[col].apply(lambda x: " ".join(x) )
    
    
    
    #a=train[[col]]
    
    #  
    words=['no free shipping','free shipping', 'free ship', '\[rm\]', 'missing']
    #replace brand
    words=a+words+words2replace
    #pattern = '|'.join(words)
    
    
    #replace used words!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
    for word in words:
        train[col]=train[col].str.replace(word, '')
        test[col]=test[col].str.replace(word, '')
    
    
    #______________________________________
    print("extra columns ", (time.clock()-t))
    
    
    #------memo---------------------
    gc.collect()
    start_mem=train.memory_usage().sum()+test.memory_usage().sum()##+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
    memory_reduce(train)
    memory_reduce(test)
    #memory_reduce(train_sp)
    #memory_reduce(test_sp)
    end_mem=train.memory_usage().sum()+test.memory_usage().sum()##+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
    print('memory change: ',(end_mem-start_mem)/start_mem)
    #-------------------------------
    
    #_____________________________________________________________________________
    '''
    /////////////////////////////////////////
                Encoding
    /////////////////////////////////////////   
    '''
    
    #brand
    #col="brand_name"
    #price_brand_med=train["price"].groupby(train[col]).median().sort_values(ascending=False)
    #col="brand_name"
    t = time.clock()
    
    #______________________________________________________
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    #???????????? new column????
    suf="_le"
    for col in ['brand_name', 'category_name_gen', 'cat_main_gen','cat_m_s1_gen']:
        train[col+suf] = le.fit_transform(train[col]).astype(int)
        dic = dict(zip(le.classes_, le.transform(le.classes_)))
        test[col+suf]=test[col].map(dic).fillna(dic["missing"]).astype(int)
        #test[col] = le.transform(test[col])
    #    test[col].fillna(value=9999, inplace=True)
    
    test[col].unique()
    
    gc.collect()
    start_mem=train.memory_usage().sum()+test.memory_usage().sum()##+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
    memory_reduce(train)
    memory_reduce(test)
    #memory_reduce(train_sp)
    #memory_reduce(test_sp)
    end_mem=train.memory_usage().sum()+test.memory_usage().sum()##+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
    print('memory change: ',(end_mem-start_mem)/start_mem)
    
    #____________________________________________________________
    
    #col='cat2_brand_qu'
    col='full_cat_brand'
    #fast code
    price_av=np.percentile(train['price'], 33)
    df = train.groupby(col)[['price']].median().astype(np.float32)
    df.columns = ['brandMedian']
    #train.drop(['brandMedian'],axis=1)
    
    train = train.join(df, on=col, how='left')
    print("Qual", rmsle(train['brandMedian'], train['price']))
    test = test.join(df, on=col, how='left').fillna(price_av)
    
    df=pd.DataFrame()
    
    
    #___________________________________________________________
    #Mean Encode
    #1
    
    extra_cat=extra_cat[:5]
    extra_col=extra_col[:5]
    
    for col in extra_col:
        for col2 in extra_cat:
            #price_av=train['price'].mean()
            price_av=np.percentile(train['price'], 33)
    #        price_av=(train['price'].mean()+train['price'].median())/2
    #        price_av=(train['price'].mean()+np.percentile(train['price'], 33))/2
    #        price_av=(np.percentile(train['price'], 50)+np.percentile(train['price'], 30))/2
    #        
            price_std=train['price'].std()
            #concat
            cols=col+'_'+col2
            train[cols]=train[col].astype(str)+train[col2].astype(str)
            test[cols]=test[col].astype(str)+test[col2].astype(str)
            
            #mean
            column_name=cols+'_mean_target'
    
    #        df=train.groupby(cols)[['price']].mean()
    #        df.columns = [column_name]
    #        
    #        train = train.join(df, on=cols, how='left')
    #        test = test.join(df, on=cols, how='left').fillna(price_av)
    
            df=train.groupby(cols)['price'].mean().astype(np.float32)
            train[column_name]=train[cols].map(df).fillna(price_av).astype(np.float32)
            test[column_name]=test[cols].map(df).fillna(price_av).astype(np.float32)
    #        
    #        print ( (train[column_name]-train[column_name+'1']).sum() )
            
    
            
            test = test.drop([cols], axis=1)
            train = train.drop([cols], axis=1)
            #del means
            df=pd.DataFrame()
            
            #val[col+'_'+col2+'mean_target']=val[col+'_'+col2].map(means)
            print("Qual "+column_name, rmsle(train[column_name], train['price']))
            
    
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
    
    
    
    print('median price time ',time.clock() - t)                                                            
    #------memo---------------------
    gc.collect()
    start_mem=train.memory_usage().sum()+test.memory_usage().sum()##+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
    memory_reduce(train)
    memory_reduce(test)
    
    end_mem=train.memory_usage().sum()+test.memory_usage().sum()##+train_sp.memory_usage().sum()+test_sp.memory_usage().sum()
    print('memory change: ',(end_mem-start_mem)/start_mem)
    #-------------------------------
    
                                                                 
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
    
    
    if nlp_yes==1:
        t=time.clock()
        #import nltk
        #nltk.download('stopwords')
        #from nltk.corpus import stopwords
        #from string import punctuation
        #from nltk.stem.porter import PorterStemmer
      
        
        # a 2 sec slower
        #t=time.clock()
        
        #print(time.clock()-t)
        
        #remove stopwords
        t=time.clock()
        #stop_words = stopwords.words('english')
        # + list(punctuation)
        ###--------------------------------
        
        col="name_desc"
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
        #tokenizing
        from sklearn.feature_extraction.text import CountVectorizer
        cv=CountVectorizer(max_features=500, 
                           ngram_range=(1,2), analyzer='word' , 
                           lowercase=False
                           , stop_words="english"
                           , min_df=0.01 ,  max_df=0.5
                           ) 
        #will leave only 1500 words
        X_train = cv.fit_transform(train[col].astype(str))
        X_train=pd.DataFrame(X_train.toarray(), columns=cv.get_feature_names())
        
        
        X_test = cv.transform(test[col].astype(str))
        X_test=pd.DataFrame(X_test.toarray(), columns=cv.get_feature_names())
        
        
        del cv
        
    #    from sklearn.feature_extraction.text import TfidfVectorizer
    #    tfidf=TfidfVectorizer(lowercase=False, ngram_range=(1,1)
    #                            , max_features=1000
    #                            , min_df=10 ,  max_df=0.98
    #                            ,stop_words="english"
    #                            )
    #    X_train = tfidf.fit_transform(train[col].astype(str)).astype(np.float32)
    #
    #    X_test = tfidf.transform(test[col].astype(str)).astype(np.float32)
    #
    #    X_test=pd.DataFrame(X_test.toarray(), columns=tfidf.get_feature_names()).to_sparse()
    #    X_train=pd.DataFrame(X_train.toarray(), columns=tfidf.get_feature_names()).to_sparse()
    #    
    #    del tfidf
        
        print("cv ",X_train.shape)
    
        memory_reduce(X_train)
    #    memory_reduce(X_test)
        gc.collect()
    
    
        print('NLP time ',time.clock() - t)  
    
    
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
    
    
    
    #Training and TEst data ---------------------------------
    print(X_train.shape , train.shape)
    X_train=pd.concat([train._get_numeric_data(), X_train.reset_index(drop=True, inplace=True)], axis=1)
    print(X_train.shape)
    X_test=pd.concat([test._get_numeric_data(), X_test.reset_index(drop=True, inplace=True)], axis=1)
    y_train=np.array(train['price'])
    
    test=pd.DataFrame()
    train=pd.DataFrame()
    #del test
    #del train
    
    X_test=X_test.drop("test_id", axis=1)
    X_train=X_train.drop("train_id", axis=1)
    X_train=X_train.drop("price", axis=1)
    
    c=set(X_test.columns)
    [x for x in X_train.columns if x not in c]
    
    
    gc.collect()
    
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
    start_mem=X_train.memory_usage().sum()+X_test.memory_usage().sum()
    memory_reduce(X_train)
    memory_reduce(X_test)
    end_mem=X_train.memory_usage().sum()+X_test.memory_usage().sum()
    print('memory change: ',(end_mem-start_mem)/start_mem)
    
    
    #--------------------------------------------------------
    #sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()
    #print('[{}] Finished to create sparse merge'.format(time.time() - start_time))
    #
    #X = sparse_merge[:nrow_train]
    #X_test = sparse_merge[nrow_train:]
    
    
    
    pattern=[]
    
    word=[]
    words=[]
    #del a
    #del b
    print('memory time ',time.clock() - t)  
    #### Model---------------------------------------------------
    
    
    #------------------------------------------------------------
    
    t=time.clock() 
    from sklearn.linear_model import Ridge, LogisticRegression
    regr_ri = Ridge( alpha=0.05,  tol=0.001,
                  max_iter=1000, normalize=True,
                  fit_intercept=True) #solver = "lsqr",
    
    import xgboost as xgb
    regr_xgb=xgb.XGBRegressor()
    
    
    
    #_____________________________________
    t=time.clock()
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split, cross_val_score
    train_X, valid_X, train_y, valid_y = train_test_split(X_train, 
                                                          y_train, 
                                                          test_size = 0.1, 
                                                          random_state = 144) 
    d_train = lgb.Dataset(train_X, label=train_y, max_bin=8192)
    d_valid = lgb.Dataset(valid_X, label=valid_y, max_bin=8192)
    watchlist = [d_train, d_valid]
    
    params = {
    	'learning_rate': 0.78,
    	'application': 'regression',
    	'max_depth': 3,
    	'num_leaves': 99,
    	'verbosity': -1,
    	'metric': 'rmsle',
    	'nthread': 4
    }
    
    
    
    
    model = lgb.train(params, train_set=d_train, num_boost_round=7500, 
                      valid_sets=watchlist, 
                      early_stopping_rounds=50, 
                      verbose_eval=500) 
    predsL = model.predict(X_test)
    
    #    Use large max_bin (may be slower)
    #    Use small learning_rate with large num_iterations
    #    Use large num_leaves (may cause over-fitting)
    #    Use bigger training data
    #    Try dart
    
    params = {
    	'learning_rate': 0.1,
    	'application': 'regression',
    	'max_depth': 5,
    	'num_leaves': 49,
    	'verbosity': -1,
    	'metric': 'rmsle',
    	'nthread': 4
    }
    
    model = lgb.train(params, train_set=d_train, num_boost_round=7500, 
                      valid_sets=watchlist, 
                      early_stopping_rounds=150, verbose_eval=500) 
    predsL2 = model.predict(X_test)
    
    del model
    print('[{}] Finished to predict lgb 1'.format(time.clock() - t))
    #------memo---------------------
    gc.collect()
    #-------------------------------
    
    
    print("Fitting Model")
    #model.fit(X_train, y_train)
    #model.coef_
    #y_pred_tr = model.predict(X_train)
    y_pred_ri,y_pred_ri_test =model_predict(regr_ri,X_train, y_train, X_test)
    
    
    print("*Error ridge", rmsle(np.round(np.expm1(y_pred_ri)) ,np.expm1(y_train)))
    
    #y_pred_xgb,y_pred_xgb_test =model_predict(regr_xgb,X_train, y_train, X_test)
    
    #print("*Error XGB ", rmsle(np.round(np.expm1(y_pred_xgb)), np.expm1(y_train)))
    #y_pred=(2*y_pred_xgb+y_pred_ri)/3
    
    
    #print("*Error combo ", rmsle(np.round(np.expm1(y_pred)), np.expm1(y_train)))
    #predict test
    
    #y_pred_test=(2*y_pred_xgb_test+y_pred_ri_test)/3
    
    #y_pred_test=(y_pred_xgb_test+y_pred_ri_test +predsL)/3
    #y_pred_test = np.exp(y_pred_test)-1
    #y_pred_test=np.expm1(y_pred_test)
    #y_pred_test=np.round(y_pred_test)
    
    #make submission
    X_test["test_id"]=test_id
    #1
    #X_test['price']=y_pred_test
    #X_test['price2']=X_test['price'].round().astype(int)
    #X_test[["test_id", "price"]].to_csv("submission.csv", index = False)
    
    #2
    #X_test['price']=np.round(y_pred_test)
    #X_test['price2']=X_test['price'].round().astype(int)
    #X_test[["test_id", "price"]].to_csv("submission_round.csv", index = False)
    
    
    #3 LGBM
    predsL=np.expm1((predsL+predsL2)/2)
    
    X_test['price']=predsL
    #X_test.drop(['price'], axis=1)
    #X_test.rename(columns={'priceLGBM': 'price'}, inplace=True)
    X_test[["test_id", "price"]].to_csv("submissionLGBM.csv", index = False)
    
    #4
    X_test['price']=np.round(predsL)
    #X_test.drop(['price'], axis=1)
    #X_test.rename(columns={'priceLGBM': 'price'}, inplace=True)
    X_test[["test_id", "price"]].to_csv("submissionLGBM_round.csv", index = False)
    
    
    print('training time ',time.clock() - t) 
    
    print('All done in ',time.clock() - start)  
    
if __name__ == '__main__':
    main()

#X_train.columns.T