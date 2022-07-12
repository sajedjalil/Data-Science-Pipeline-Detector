# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 01:37:27 2016
make for bank satisfaction analysis
@author: root
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.feature_selection import SelectPercentile
from sklearn.grid_search import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
import xgboost as xgb

from sklearn.decomposition import PCA
"""
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
"""

def delete_cons(train,test):
    train_std=np.std(train,axis=0)
    test_std=np.std(test,axis=0)
    ind=[]
    for i in range(0,train.shape[1]):
        if train_std[i]==0 and test_std[i]==0:
           ind.append(i)
    train=np.delete(train,ind,1) 
    test=np.delete(test,ind,1)      
    return train,test   
    
def sort_feature(raw_train,raw_test):
    #create sorting  sequence feature by each row
    #raw_features should be a  np.narray
    sort_train=np.argsort(raw_train,axis=1)
    sort_test=np.argsort(raw_test,axis=1)
    return sort_train,sort_test
    
def sparse_feature(raw_features):
    #create sparse feature by each columns
    #each values range in [1,10]
    #1 represent raw values belong to the smallest 10% of (MAX-MIN)
    #raw_features should be a np.narray
    spar_f=raw_features
    for col in range(0,raw_features.shape[1]):
        MAX_MIN=max(raw_features[:,col])-min(raw_features[:,col])
        step_size=MAX_MIN/10.0
        for i in range(0,10,1):
            low_thredhold=step_size*i
            up_thredhold=step_size*(i+1)
            if i==0:
                index=(raw_features[:,col]<= up_thredhold)
                spar_f[:,col][index]=i+1
            elif i==9:
                index=(raw_features[:,col]>low_thredhold)
                spar_f[:,col][index]=i+1
            else:
                index=np.logical_and(raw_features[:,col]>low_thredhold,raw_features[:,col]<= up_thredhold)
                spar_f[:,col][index]=i+1 

    return spar_f
   
def count_sparse(raw_features):
    #print"Generating count of sparse feature:"
    sum_sparse=sp.empty((raw_features.shape[0],1))
    for j in range(1,11,1):
        condition=(raw_features==j)
        temp=np.sum(condition,axis=1)
        temp=np.reshape(temp,(raw_features.shape[0],1))
        sum_sparse=sp.column_stack((sum_sparse, temp))
        
    sum_sparse=np.delete(sum_sparse,0,axis=1) 
    return sum_sparse
      
def feature_generator(X_train,X_test):   
    #generate features on the base of raw features (x_train and x_test)
    #create sort sequence and sparse features on training data
    print("Generating sort sequence feature:")
    train_seq,test_seq=sort_feature(X_train,X_test)
    
    #create sparse feature on numeric features
    print("Generating sparse feature:")
    print(" # each values range in [1,10]")
    print(" #1 represent raw values belong to the smallest 10% of (MAX-MIN)")
    train_spar=sparse_feature(X_train)
    test_spar=sparse_feature(X_test)
    #delete constant values in sparse features            
    train_spar,test_spar=delete_cons(train_spar,test_spar)
    
    #sum of sparse feature on each row
    print("Generating count of sparse feature:")
    c_train_spar=count_sparse(train_spar)
    c_test_spar=count_sparse(test_spar) 
    #delete constant values in count sparse features            
    c_train_spar,c_test_spar=delete_cons(c_train_spar,c_test_spar)
    
    return  train_seq,test_seq, train_spar, test_spar,c_train_spar,c_test_spar
    
def feature_PCA(sort_seq,spar_f,sum_sparse,raw_features):
    #original shape of sort and sparse features 
    num_seq=sort_seq.shape[1]
    num_spar=spar_f.shape[1]
    print("Reserving 25 percent of %d features from sort_feature and raw features by PCA" %num_seq)
    #pca reserve 25% features on sort_seq and raw_features
    pca=PCA(n_components=int(num_seq/4))
    pca_sort=pca.fit_transform(sort_seq)
    pca_raw_features=pca.fit_transform(raw_features)
    #pca2 reserve 25% features on sparse features
    pca2=PCA(n_components=int(num_spar/4))
    pca_spar_f=pca2.fit_transform(spar_f)
    #pca3 reserve 2 feaeture on sum_sparse feature
    pca3=PCA(n_components=2)
    pca2_sum_sparse=pca3.fit_transform(sum_sparse)
    #stack them together
    pca_infu=sp.column_stack((pca_sort,pca_spar_f,pca2_sum_sparse,pca_raw_features))
    return pca_infu
    
def select_from_model(features,target):

    #select the best 25% of the features
    clf =SelectPercentile(percentile=25)
    new_f=clf.fit_transform(features, target)
    selected_ind=clf.get_support(indices=True)
    n_features =new_f.shape[1]   
    print ("   select %d features  by linear model"%n_features)
    return new_f,selected_ind
   
def model_select_all(X_train,X_test,train_seq,test_seq, train_spar, test_spar,c_spar_train,c_spar_test,target):
    print("SelectPercentile model selection on sort features:")
    new_sort_train,selected_ind0=select_from_model(train_seq,target)
    new_sort_test=test_seq[:,selected_ind0]
    print("SelectPercentile model selection on sparse features:")
    new_spar_train,selected_ind1=select_from_model(train_spar,target)
    new_spar_test=test_spar[:,selected_ind1] 
    print("SelectPercentile model selection on count sparse features:")
    new_c_spar_train,selected_ind2=select_from_model(c_spar_train,target)
    new_c_spar_test=c_spar_test[:,selected_ind2]  
    print("SelectPercentile model selection on raw_features features:")
    new_raw_train,selected_ind3=select_from_model(X_train,target) 
    new_raw_test=X_test[:,selected_ind3]
    
    #stack them together
    model_train=sp.column_stack((new_sort_train,new_spar_train,new_c_spar_train,new_raw_train))
    model_test=sp.column_stack((new_sort_test,new_spar_test,new_c_spar_test,new_raw_test))
    return model_train,model_test
    
def load_data():
    # load data and initialize y and x from train set and test set
    df_train = pd.read_csv('../input/train.csv')
    df_test = pd.read_csv('../input/test.csv')
    
    target = df_train['TARGET'].values
    df_train =df_train.drop(['ID','TARGET'],axis=1)
    id_test = df_test['ID']
    df_test = df_test.drop(['ID'],axis=1)
    
    # check whether features are arrange the same as in train set and test set
    list_train=df_train.columns.tolist()
    list_test =df_test.columns.tolist()
    print( 'Checking consistence of features from different sets is :',list_train==list_test)
    
    # remove constant or zero columns in train set and test set
    cons_col = []
    for col in df_train.columns:
        if df_train[col].std() == 0 :
            #add ind in to remove_ind
            cons_col.append(col)
    
    df_train.drop(cons_col, axis=1, inplace=True)
    df_test.drop(cons_col, axis=1, inplace=True)
    
    print ('deleting %d columns with zeros or constant values  from train set and test set' %len(cons_col))
    
    # remove duplicated columns
    indent_col = []
    #number of columns in df_train
    c = df_train.columns
    for i in range(len(c)-1):
        v1_train = df_train[c[i]].values
        for j in range(i+1,len(c)):
            v2_train = df_train[c[j]].values
            if np.array_equal(v1_train,v2_train):
                indent_col.append(c[j])
    
    df_train.drop(indent_col, axis=1, inplace=True)
    df_test.drop(indent_col, axis=1, inplace=True)
    print ('deleting %d columns with identical values from train set and test set' %len(indent_col))
    """
    # delete columns that highly correlated with R>0.99 
    corr_col = []
    c = df_train.columns
    R_train = np.corrcoef(df_train.T)
    R_test = np.corrcoef(df_test.T)
    for i in range(len(c)-1):
        for j in range(i+1,len(c)):
            if R_train[i,j]>0.99 and R_test[i,j]>0.99:
                corr_col.append(c[j])    
                            
    df_train.drop(corr_col, axis=1, inplace=True)
    df_test.drop(corr_col, axis=1, inplace=True)
    print ('deleting %d columns with correration factor >0.99 from train set and test set \n') % (len(corr_col))
    """
    
    feature_names=df_train.columns.values.tolist()
    X_train=df_train.values
    X_test = df_test.values
    #copy train and  test data into temperory data
    temp_train=np.copy(X_train)
    temp_test=np.copy(X_test)
    train_seq,test_seq, train_spar, test_spar,c_spar_train,c_spar_test=feature_generator(temp_train,temp_test)
    return  X_train,X_test,id_test,feature_names,train_seq,test_seq,train_spar,test_spar,target,c_spar_train,c_spar_test

    
def models_test(train,target):
    print("spliting data into two sets:fitting and evaluation")
    """
    # data normalization
    scaler=StandardScaler()
    train = scaler.fit_transform(train) 
    """
    X_fit, X_eval, y_fit, y_eval= train_test_split(train, target, test_size=0.3) 
    num_features=X_fit.shape[1]
    print("")
    print("Starting training with %d number of features....."%num_features)
    
    #train 
    xgb_clf=train_xgb(X_fit,y_fit,X_eval,y_eval)
    #svm_clf=train_svm(X_fit,y_fit,X_eval,y_eval)
    #rf_clf=train_randomforest(X_fit,y_fit,X_eval,y_eval)
    #et_clf=train_etratree(X_fit,y_fit,X_eval,y_eval)
    #ada_clf=train_adaboost(X_fit,y_fit,X_eval,y_eval)
    #nn_clf=train_nn(X_fit,y_fit,X_eval,y_eval,num_features)
    
    return xgb_clf
    
def model_auc(model,X_eval,y_eval):
    pred_eval=model.predict_proba(X_eval)[:,1]  
    eval_auc=roc_auc_score(y_eval,pred_eval)
    print ("AUC of the model is :  ",eval_auc )
    print("")
    
def train_nn(X_fit,y_fit,X_eval,y_eval,num_features):
    print("training data with neuralnetwork model")
    batch_size = 500
    #max iteration
    max_epoch = 10

    clf = Sequential()
    clf.add(Dense(400, input_shape=(num_features,)))
    clf.add(Activation('tanh'))
    clf.add(Dropout(0.2))
    clf.add(Dense(100))
    clf.add(Activation('tanh'))
    clf.add(Dense(1))
    clf.add(Activation('sigmoid'))
    
    adam=Adam(lr=0.03, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    clf.compile(loss='binary_crossentropy',
              optimizer=adam)
    
    clf.fit(X_fit, y_fit,
              batch_size=batch_size, nb_epoch= max_epoch,
              show_accuracy=True, verbose=0,
              validation_data=(X_eval, y_eval))
    pred_eval=clf.predict_proba(X_eval) 
    eval_auc=roc_auc_score(y_eval,pred_eval)
    print ("AUC of the model is :  ",eval_auc )
    print("")
    return clf    
    
def train_adaboost(X_fit,y_fit,X_eval,y_eval):
    print("training data with AdaBoostClassifier model")
    clf=AdaBoostClassifier(n_estimators=50,learning_rate=0.1)
    clf.fit(X_fit, y_fit)
    model_auc(clf,X_eval,y_eval)
    return clf
    
def train_etratree(X_fit,y_fit,X_eval,y_eval):
    print("training data with ExtraTreesClassifier model")
    clf=ExtraTreesClassifier()
    clf.fit(X_fit, y_fit)
    model_auc(clf,X_eval,y_eval)
    return clf
    
def show_all_options(options):
    print ('')
    print ('ALL options :')
    print ( '  max_depth:        ',options[0],'     n_esimatiors:     ',options[1])
    print ( '  learning rate:    ',options[2],'  subsample:        ',options[3])
    print ('  colsample_bytree: ',options[4],'  seed:             ',options[5])
    print('')
    
def train_xgb(X_fit,y_fit,X_eval,y_eval):
    print ('Starting training by xgboost method with options .......\n' )
    # set option for xgboost
    #options=[5,350,0.03,0.95,0.85,4242] 
    #show_all_options(options)
    # XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242)
    #initial classifier of xgboost

    model = xgb.XGBClassifier(missing=None,learning_rate=0.03,subsample=0.95,nthread=4,
                              colsample_bytree=0.85, seed=4242,silent=True)
    
    clf = GridSearchCV(model,
               {"max_depth": [5],
                "n_estimators": [225,250,275]}, 
                 scoring="roc_auc",  
                 n_jobs=-1,
                 verbose=1)

    # fitting
    clf.fit(X_fit, y_fit)
    print(clf.best_score_)
    print(clf.best_params_)
    """
    clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242)    
     # fitting
    clf.fit(X_fit, y_fit, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])
    """    
    
    #clf.fit(X_fit, y_fit, early_stopping_rounds=10, eval_metric="auc", eval_set=[(X_eval, y_eval)],verbose=1)
    #print (clf.best_score_, clf.best_params_)
    auc_eval=roc_auc_score(y_eval, clf.predict_proba(X_eval)[:,1])
    auc_fit=roc_auc_score(y_fit, clf.predict_proba(X_fit)[:,1])
    overfit_rate=auc_fit/auc_eval
    print ('')
    print ('   AUC of fitting data:',auc_fit)
    print ('   AUC of cross validation :',auc_eval)
    print ('   Overfitting rate  :',overfit_rate)
    print ('')
    return clf    

    
def train_randomforest(X_fit,y_fit,X_eval,y_eval):
    print("training data with RandomForestClassifier model")
    clf=RandomForestClassifier()
    clf.fit(X_fit, y_fit)
    model_auc(clf,X_eval,y_eval)
    return clf
    
def train_svm(X_fit,y_fit,X_eval,y_eval):
    print("training data with linear_svm model")
    clf=SVC(C=1.0, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=50, probability=True, random_state=None, shrinking=True,
    tol=0.01, verbose=False)
    clf.fit(X_fit, y_fit,)
    model_auc(clf,X_eval,y_eval)
    return clf    
    
        
if __name__=='__main__':
    
    X_train,X_test,id_test,feature_names,train_seq,test_seq,train_spar,test_spar,target,c_spar_train,c_spar_test=load_data()
    print("" )
    #test models base on raw_features,sort_feature,sparse_features and count_sparse_features
    print( "========== test model on raw features ============")
    xgb_clf1=models_test(X_train,target)
    """
    print "========== test model on sort sequence features =="
    models_test(train_seq,target)
    print "========== test model on sparse features ========="
    models_test(train_spar,target)
    print "========== test model on count sparse features ==="
    models_test(c_spar_train,target)
   
    
    #test models on infusion features by PCA or model_selection
    pca_infu=feature_PCA(train_seq,train_spar,c_spar_train,X_train)
    print "========== test model on pca infusion features :======="
    models_test(pca_infu,target)
          
    model_s_train,model_s_test=model_select_all(X_train,X_test,train_seq,test_seq, train_spar, test_spar,c_spar_train,c_spar_test,target)
    print ("========== test model on model infusion features:=======")
    xgb_clf2=models_test(model_s_train,target) 
    
    
    # predicting
    # model_infu_test=model_select_all(test_seq,test_spar,c_spar_test,X_test,target)
    y_pred_1=xgb_clf1.predict_proba(X_test)[:,1]
    y_pred_2=xgb_clf2.predict_proba(model_s_test)[:,1]
    R_of_pred=np.corrcoef(y_pred_1,y_pred_2)
    print ("relative of y_pred_1 and y_pred_2",R_of_pred[0,1])
    y_pred=(y_pred_1+y_pred_2)/2
    submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})
    """