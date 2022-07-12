# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 16:58:23 2016

@author: Huster-V3
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import theano
import theano.tensor as T
import lasagne

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
    
def drop_data(train,test,drop_list):
    train = train.drop(drop_list,axis=1)
    test = test.drop(drop_list,axis=1)
    return train,test
    

        
def load_data():
    print('Loading data...')
    # load train data    
    df_train = pd.read_csv("../input/train.csv")
    target = df_train['target'].values
    df_train =df_train.drop(['target'],axis=1)
    # load test data
    df_test = pd.read_csv("../input/test.csv")
    id_test = df_test['ID'].values
    #train.drop(labels = ["ID","target","v107","v71","v100","v63","v64"], axis = 1, inplace = True)
    df_train.drop(['v8','v23','v25','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1, inplace = True)
    
    #test.drop(labels = ["ID","v107","v71","v100","v63","v64"], axis = 1, inplace = True)
    df_test.drop(labels = ['v8','v23','v25','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1, inplace = True)
    
    print(' find categorical variables from different columns ')
    #find categorical variables
    categoricalVariables = []
    for var in df_train.columns:
        vector=pd.concat([df_train[var],df_test[var]], axis=0)
        typ=str(df_train[var].dtype)
        if (typ=='object'):
            categoricalVariables.append(var)
    print(' categorical variables is ', categoricalVariables)
       
    print('Converting categorical values into numbers of lables')
    
    for col in categoricalVariables:
        df_train[col] = pd.factorize(df_train[col])[0] 
        df_test[col] = pd.factorize(df_test[col])[0]   
                                  
    #check whether features are arrange the same as in train set and test set
    list_train=df_train.columns.tolist()
    list_test =df_test.columns.tolist()
    print ('Checking consistence of features from different sets is :',list_train==list_test)

    #Remove sparse columns
    sparse_col=[]
    for col in df_train.columns:
        cls=df_train[col].values
        if sum(cls)<10:
           sparse_col.append(col) 
           
    df_train.drop(sparse_col, axis=1,inplace=True)
    df_test.drop(sparse_col, axis=1,inplace=True)
    print ('deleting %d columns with sparse values  from train set and test set'% len(sparse_col))
 
    df_train=df_train.fillna(-1)        
    df_test=df_test.fillna(-1)   
        
    feature_names=df_train.columns.values.tolist()
    print ('Assigning values in to X_train and X_test......................................\n')
    X_train =df_train.values
    X_test = df_test.values

    print ('training insurance claims data with %d features  from train set ' %len(feature_names))
   
    print ('Spliting train data into three parts:fit,eval.............\n' )
    scaler=StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test=scaler.fit_transform(X_test)
    X_fit, X_eval, y_fit, y_eval= train_test_split(X_train, target, test_size=0.2,random_state=100)
    num_features=len(feature_names)
    return X_fit, X_eval, y_fit, y_eval,X_test,id_test,num_features
    
if __name__ == "__main__":     
    
    X_fit, X_eval, y_fit, y_eval,X_test,id_test,num_features=load_data()
    
    batch_size = 500
    #max iteration
    max_epoch = 100

    model = Sequential()
    model.add(Dense(50, input_shape=(num_features,)))
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(40))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    adam=Adam(lr=0.03, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='binary_crossentropy',
              optimizer="rmsprop")
    
    model.fit(X_fit, y_fit,
              batch_size=batch_size, nb_epoch= max_epoch,
              show_accuracy=True, verbose=2,
              validation_data=(X_eval, y_eval))
    score = model.evaluate(X_eval, y_eval,
                           show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
   
    y_pred = model.predict_proba(X_test)[:,0]
   
    print ('writting predicted results into random_forest.csv')
    submission = pd.DataFrame({"ID":id_test, "PredictedProb":y_pred})
    submission.to_csv('keras_nn.csv', index=False)
    