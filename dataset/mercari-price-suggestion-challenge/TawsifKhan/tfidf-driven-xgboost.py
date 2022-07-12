# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 18:26:50 2018

@author: tawsif.khan
"""

#Load packages
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import re,gc,time
from xgboost import XGBRegressor
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from multiprocessing import Pool
from matplotlib import pyplot as plt
from scipy.sparse import hstack, csr_matrix


#Eval funcs
def rmsle(y, y0):
    assert len(y) == len(y0)
    try:
        error = np.sqrt(np.nansum(np.square(np.log(y0 + 1) - np.log(y + 1)))/float(len(y)))
        return(error)
    except:
        return(1)
    
def rmse(y, y0):
    assert len(y) == len(y0)
    try:
        error = np.sqrt(np.nansum(np.square(y0 - y))/float(len(y)))
        return(error)
    except:
        return(1)

#Data cleaning funcs
def fill_na(df):
        df.category_name.fillna(value="missing",inplace=True)
        df.brand_name.fillna(value="missing",inplace=True)
        return(df)

def clean_text_apply(df):
        df['ndesc']=df['ndesc'].apply(clean_text)
        return(df)
        
def clean_text(text):
    stemmer = PorterStemmer()
    try:
        rm_reg_feat=re.findall("[a-z0-9a-z]+",text.lower())
        stemmed = [stemmer.stem(item) for item in rm_reg_feat]
        return(" ".join(stemmed))
    except:
        return("")
        
#DF parallelizer
def parallelize_df(df,func):
        pool = Pool(processes=4)
        a,b,c,d = np.array_split(df,4)
        df=pd.concat(pool.map(func,[a,b,c,d]))     
        pool.close()
        pool.join()
        return(df)


#Main 
if __name__ == '__main__':
    time_ = time.time()
    low_data_volume = 1 #1 - Load small portion of data to test code 
                        #0 - Load all data
    dev_mode = 3    #1- Grid search 
                    #2-Train only validation. 
                    #3-Kaggle submission
    f_n = 50000 #Number of features to be collected
    
    #Load data
    print("\nMercari Price Suggestion Challenge")
    print("\nStep 1 of 6: Load Data")
    train = pd.read_table("../input/train.tsv")
    test = pd.read_table("../input/test.tsv")
    if low_data_volume == 1:
        train = train.sample(n=100000,random_state=123)
        test = test.sample(n=1000,random_state=123)
        f_n = 2000
    
    submission = pd.DataFrame(test.test_id)
    
    
      
    #Clean data
    print("\nStep 2 of 6: Clean data")
    train = fill_na(train)
    test = fill_na(test)
    #Item description has it's own id for missing info.
    #but reducing the number of words
    train.loc[train['item_description'] == "No description yet", 'item_description'] = "missing"
    test.loc[test['item_description'] == "No description yet",'item_description'] = "missing"
    #Concatenate name and item_description
    #The corpus for TFIDF will be built off of this
    train['ndesc'] = train['name'] + str(" ") + train['item_description']
    test['ndesc'] = test['name'] + str(" ") +  test['item_description']
    train = train.drop(['name','item_description'],axis=1)
    test = test.drop(['name','item_description'],axis=1)
    
    #Clean the name and item description
    #Multiprocessing
    
    train=parallelize_df(train,clean_text_apply)
    test=parallelize_df(test,clean_text_apply)
    print('\nText cleaning complete. Total time elapsed: ' + str(int(time.time()-time_)) + 's')
    
    #Convert categorical variables into numerical 
    print("Step 3 of 6: Convert categorical string variables to numerical")
    le = preprocessing.LabelEncoder()
    le.fit(np.hstack([train.category_name, test.category_name]))
    
    train.category_name = le.transform(train.category_name)
    test.category_name = le.transform(test.category_name)
    
    le.fit(np.hstack([train.brand_name, test.brand_name]))
    train.brand_name = le.transform(train.brand_name)
    test.brand_name = le.transform(test.brand_name)
    
    #TFIDF weighitng of each word and select top f_n words as features
    #Extrac features from name and item_description
    
    print('Step 4 of 6: Apply TFIDF Vectorizer.\nExtract ' + str(f_n) + ' features.')
    
    corpus=np.hstack([train.ndesc,test.ndesc])
    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, max_df=.99,
                                    use_idf=True, smooth_idf=False,
                                    stop_words='english',
                                    sublinear_tf=True,max_features=f_n)
    transformed = sklearn_tfidf.fit_transform(corpus)
    
    #Drop the description columns after transformation
    test = test.drop(['ndesc'],axis=1)
    train = train.drop(['ndesc'],axis=1)
    
    gc.collect()
    
   #print('Name and description keywords: ' + str(ndesc_feat))
   
    print("Features collected: " + str(len(sklearn_tfidf.get_feature_names())) + ". Time elapsed: " + str(int(time.time()-time_)) + "s")
    print("\nStep 5 of 6: Prep input for model: dataframe to sparse.")
    #Modelling
    
    y = train['price'].apply(np.log1p)
    train = train.drop(['price'],axis=1)
    
    train = csr_matrix(train.values)
    train = hstack([train,transformed[0:train.shape[0]]])
    
    test = csr_matrix(test.values)
    test = hstack([test,transformed[train.shape[0]:]])
    
    X_train, X_valid, y_train, y_valid = train_test_split(train, y, test_size=0.20, random_state=123)
    
    del train
    gc.collect()
    
    print("\nStep 6 of 6: Model and predict.")
    if dev_mode in [2,3]:
        print("Creating predictor.")
        xgb_model = XGBRegressor(n_estimators=10
                                    ,max_depth = 38
                                    ,min_child_weight = 11
                                    ,n_jobs = 4
                                    ,subsample = 0.8
                                    )
        xgb_model.fit(X_train,y_train,
                     eval_set=[(X_train, y_train), 
                               (X_valid, y_valid)]
                       ,early_stopping_rounds=200
                    )
        
        print("Modeling complete. Time elapsed: " + str(int(time.time()-time_)) + "s")
        y_pred = xgb_model.predict(X_valid)
        y_pred = np.exp(y_pred)-1
        
        y_true = np.exp(y_valid)-1
        
        v_rmsle = rmsle(y_true, y_pred)
        print("\nRMSLE xg: "+str(v_rmsle))
        x=range(int(np.amax(y_true)+1))
        y=np.repeat(0,len(x))
        plt.scatter(y_true,y_pred, marker='o', color='r')
        plt.plot(x,x,color='b')
        plt.plot(x,y,color='b')
        
        plt.show()
    
    if dev_mode == 3:
        print("\nPredict and submit.")
        y_pred = xgb_model.predict(test)
        y_pred = np.exp(y_pred)-1
        submission['price'] = y_pred
        submission.to_csv("output_price.csv",index=False)
        
    if dev_mode == 1:   
        gc.collect()
        print("Initiating grid search")
        xgb_model = XGBRegressor(n_jobs=4)
        param_grid = { 
                   #"n_estimators" : [1000],
                   "max_depth" : [10,14,20], #17-11
                   "min_child_weight" : [11],
                   "subsample":[ .8]
                   #"gamma":[0,0.1,0.2,0.5]
                   #"learning_rate":[.055,.060,.065]
                   #"subsample" :[i/10.0 for i in [2,4,8]]
                   }
        CV_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid,verbose=1)
        CV_xgb.fit(X_train, y_train)
        print(CV_xgb.best_params_,CV_xgb.best_score_) 
        print(getattr(CV_xgb,'grid_scores_',None))
    print("All tasks complete.")