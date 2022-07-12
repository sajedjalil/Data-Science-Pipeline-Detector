# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#%env JOBLIB_TEMP_FOLDER=/tmp
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input/home-data"))

# Any results you write to the current directory are saved as output.

# -*- coding: utf-8 -*-
"""
Created on Tue May 29 16:48:42 2018

@author: Yury
"""

import os
os.environ['MKL_NUM_THREADS'] = '4' 
os.environ['OMP_NUM_THREADS'] = '4'
    

import numpy as np
import pandas as pd
import time
st=time.clock()



from sklearn.metrics import roc_auc_score


import gc





def reduce_mem_usage(dataset):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = dataset.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    i=1
    for col in dataset.columns:
        col_type = dataset[col].dtype
        
        if col_type != object:
            c_min = dataset[col].min()
            c_max = dataset[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dataset[col] = dataset[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dataset[col] = dataset[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dataset[col] = dataset[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dataset[col] = dataset[col].astype(np.int64)  
                #end_mem = dataset.memory_usage().sum() / 1024**2
                #print(i,' Memory usage after optimization is: {:.2f} MB'.format(end_mem))
                gc.collect()
                i+=1
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    dataset[col] = dataset[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    dataset[col] = dataset[col].astype(np.float32)
                else:
                    dataset[col] = dataset[col].astype(np.float64)
                #end_mem = dataset.memory_usage().sum() / 1024**2
                #print(i, 'Memory usage after optimization float is: {:.2f} MB'.format(end_mem))
                gc.collect()
                i+=1
    end_mem = dataset.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return dataset



def merge_l(a, b, c):
    ind=a.index
    a=a.merge(b, on=c, how='left')
    a.index=ind
    return a

def handle_missing(dataset, col, val):
    dataset[col].fillna(value=val, inplace=True)
    return (dataset)

'''---------------------------------------------'''
import dask.dataframe as dd

#df_train = pd.read_csv('../data/train_y.csv',  engine="python") #index_col='zone_id'
#df_test = pd.read_csv('../data/test_y.csv',  engine="python") #index_col='zone_id'
#df_buro = pd.read_csv('../data/buro.csv',  engine="python") #index_col='zone_id'


df_train = dd.read_csv('../input/home-allin/train_allin.csv')
df_test = dd.read_csv('../input/home-allin/test_allin.csv')
df_test=df_test.compute()
df_train=df_train.compute()





    
gc.collect()
'''-----------------------------------------------------------------------'''


'''-----------------------------------------------------------------------'''
y_target=df_train['TARGET']
ids_tr=df_train['SK_ID_CURR']
ids_te=df_test['SK_ID_CURR']

col_drop=['SK_ID_CURR', 'TARGET']

for c in col_drop:
    df_train.drop([c], inplace=True, axis=1)
    
  
df_test.drop(['SK_ID_CURR'], inplace=True, axis=1)

#DO we need to drop ID??????????????????????????????


print('1')

'''-----------------------------------------------------------------------'''
#df_train= memory_reduce(df_train)
#df_test= memory_reduce(df_test)

#df_train=reduce_mem_usage(df_train)
#df_test=reduce_mem_usage(df_test)

#df_s=df_train.head(100)
'''-----------------------------------------------------------------------'''


gc.collect()
'''-----------------------------------------------------------------------'''


folds=10
rep=1
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=121)


#
#from sklearn.model_selection import RepeatedKFold
#
#skf = RepeatedKFold(n_splits=folds,  n_repeats=rep, random_state=121)
#from sklearn.model_selection import RepeatedStratifiedKFold
#skf = RepeatedStratifiedKFold(n_splits=folds,  n_repeats=rep, random_state=12121,
#                              indices=True)



from xgboost              import XGBClassifier
from sklearn.ensemble     import ExtraTreesClassifier
from sklearn.tree         import ExtraTreeClassifier
from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import GradientBoostingClassifier
from sklearn.ensemble     import BaggingClassifier
from sklearn.ensemble     import AdaBoostClassifier
from sklearn.ensemble     import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm             import LGBMClassifier


clfs = {
        'xgb_cl':XGBClassifier(seed=111, 
                               n_estimators=500,
                               max_depth=6, 
                               objective='binary:logistic',
                               nthread=4,
                               eval_metric='auc') #, n_jobs=-1
        ,              
        'lgbm_cl':LGBMClassifier(n_estimators=500, 
                                 silent=True,
                                 nthread=4,
#                                 n_estimators=10000,
                                 learning_rate=0.02,
                                 num_leaves=40,
                                 max_depth=8,
#                                boosting_type='dart',
                                 objective='binary')
#,        
##        # criterion = 'gini' 'entropy'
##        'exttr':ExtraTreesClassifier(n_estimators = 150, criterion = 'gini', random_state = 100),
###        'bag':BaggingClassifier(),          
###        'decis':DecisionTreeClassifier(),
#        'gradboost':GradientBoostingClassifier(random_state=111,n_estimators=200, 
#                                               learning_rate=0.1,
#                                               max_depth =3)
##, 
##        'logist':LogisticRegression(),
##        'ada':AdaBoostClassifier(base_estimator= ExtraTreesClassifier(),
##                                random_state=111,n_estimators=200, 
##                                               learning_rate=0.1),         
#        'forest':RandomForestClassifier(n_estimators = 200, criterion = 'gini', 
#                                        random_state = 100, n_jobs=-1)
        }

print('to training')
spars=1
if spars==1:
    #---------SPARSE----------------------
    from scipy.sparse import csr_matrix
    df_train_s=csr_matrix(df_train.fillna(0)).tocsr(copy=False)
    df_test_s=csr_matrix(df_test.fillna(0)).tocsr(copy=False)
    
    print('sparse finished')
    #df_train_s=csr_matrix(df_train.fillna(0)).tocsr(copy=False)
    #df_test_s=csr_matrix(df_test.fillna(0)).tocsr(copy=False)
    #y_target_s=csr_matrix(y_target.fillna(0)).tocsr(copy=False)
    
    #df_train_s = df_train.fillna(0).to_sparse(fill_value=0).tocsr()
    #df_test_s = df_test.fillna(0).to_sparse(fill_value=0).tocsr()


singl=0
if singl==1:
    df_train_predictions=pd.DataFrame(index=ids_tr)
    df_test_predictions=pd.DataFrame(index=ids_te)
    
    for c in list(clfs.keys()):
        classifier=clfs[c]
        av_score=0
        y_pred_fl_tr=np.zeros(df_train.shape[0])
        y_pred_fl_te=np.zeros(df_test.shape[0])
    #    for train_index, test_index in skf.split(df_train.index, y_target):
        for train_index, test_index in skf.split(df_train, y_target):
            if spars==1:
                #fit
                train_s=df_train.iloc[train_index].fillna(0).to_sparse(fill_value=0).to_csr()
                classifier.fit(train_s, np.array(y_target.iloc[train_index]))
                
                #valid
                test_s=df_train.iloc[test_index].fillna(0).to_sparse(fill_value=0).to_csr()
                y_pred = classifier.predict_proba(test_s)
                av_score+=roc_auc_score(y_target.iloc[test_index], y_pred[:,1])
                print("Qual ", c," : " , roc_auc_score(y_target.iloc[test_index], y_pred[:,1]))
                #predict
                y_pred_fl_te+=classifier.predict_proba(df_test_s)[:,1]
                y_pred_fl_tr+=classifier.predict_proba(df_train_s)[:,1]
                
            else:
                
                #fit    
                classifier.fit(df_train.iloc[train_index], np.array(y_target.iloc[train_index]))
                #valid      
                y_pred = classifier.predict_proba(df_train.iloc[test_index])
                av_score+=roc_auc_score(y_target.iloc[test_index], y_pred[:,1])
                print("Qual ", c," : " , roc_auc_score(y_target.iloc[test_index], y_pred[:,1]))
                #predict
                y_pred_fl_te+=classifier.predict_proba(df_test)[:,1]
                y_pred_fl_tr+=classifier.predict_proba(df_train)[:,1]
                
                
        print(c, "Av_Qual: " , av_score/(folds*rep))
        y_pred_fl_tr/=(folds*rep)
        y_pred_fl_te/=(folds*rep)
        df_train_predictions[c]=y_pred_fl_tr
        df_test_predictions[c]=y_pred_fl_te
        if c=='lgbm_cl':
            a=classifier.feature_importances_
            a=pd.DataFrame(a, index=df_train.columns)





#voting
#df_train_predictions=pd.DataFrame(index=df_train.index)
#df_test_predictions=pd.DataFrame(index=df_test.index)

from sklearn.ensemble import VotingClassifier

vc=VotingClassifier(clfs.items(), voting='soft')

y_pred_fl_te=np.zeros(df_test.shape[0])


av_score=0

for train_index, test_index in skf.split(df_train.index, y_target):
    if spars==1:
        #fit
        print('fit')
        vc.fit(csr_matrix(df_train.iloc[train_index].fillna(0)).tocsr(copy=False), np.array(y_target.iloc[train_index]))
                
        #valid
        print('valid')
        y_pred = vc.predict_proba(csr_matrix(df_train.iloc[test_index].fillna(0)).tocsr(copy=False))
        av_score+=roc_auc_score(y_target.iloc[test_index], y_pred[:,1])
        print("Qual: " , roc_auc_score(y_target.iloc[test_index], y_pred[:,1]))
        
        #predict
        print('predict')
        y_pred_fl_te+=vc.predict_proba(csr_matrix(df_test.fillna(0)).tocsr(copy=False))[:,1]
        
    else:
        vc.fit(df_train.iloc[train_index], np.array(y_target.iloc[train_index]))
        y_pred = vc.predict_proba(df_train.iloc[test_index])
        av_score+=roc_auc_score(y_target.iloc[test_index], y_pred[:,1])
        print("Qual: " , roc_auc_score(y_target.iloc[test_index], y_pred[:,1]))
        y_pred_fl_te+=vc.predict_proba(df_test)[:,1]
    gc.collect()




print(c, "Av_Qual: " , av_score/(folds*rep))
y_pred_fl_te/=(folds*rep)

#delete frame and release memory
lst = [pd.DataFrame(), df_train_s]
#del a, b, c # dfs still in list
del lst     # memory release now


#---------------------------------------------------------------------------
df_subm=pd.DataFrame(index=ids_te)
df_subm['SK_ID_CURR'] = df_subm.index
df_subm['TARGET'] = y_pred_fl_te

#---------------------------------------------------------------------------
import datetime

td=datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")


df_subm.to_csv('submission_'+ '_'+ td+'.csv', index=False)



