# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.interpolate import interp1d
import math
import scipy as sc
from pandas.compat.numpy import*
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw, show, plot
import sklearn
from multiprocessing import Process
#from sklearn.datasets import load_boston
from sklearn.linear_model import*
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error,explained_variance_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import svm, datasets
from sklearn.utils.multiclass import check_classification_targets
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
####
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
Elo_train           =pd.read_csv("../input/train.csv")
print (Elo_train.head(5))
#Elo_merch           =pd.read_csv("merchants.csv")
#Elo_hist            =pd.read_csv("historical_transactions.csv")
#Elo_newMerch        =pd.read_csv("new_merchant_transactions.csv")
Elo_test            =pd.read_csv("../input/test.csv")
Elo_train_header    =list(Elo_train)#Elo.columns
Elo_test_header     =list(Elo_test)
###### DATA HEAD

print (Elo_test.head(5))
print ('Train shape', Elo_train.shape)
print ('Test shape', Elo_test.shape)
print ('Train data isnull sum', Elo_train.isnull().sum())
print ('Test data isnull sum', Elo_test.isnull().sum())
####### DROPPING NAN DATA
####Elo_test.dropna(axis=0,inplace =True)  ## Dropping missing value
print ('Train data isnull sum', Elo_train.isnull().sum())
print ('Test data isnull sum', Elo_test.isnull().sum())
print ('shape of test data', Elo_test.shape)
########## 
Elo_train_nbreRows  =Elo_train.shape
Elo_test_nbreRows   =Elo_test.shape
Elo_train_target    =Elo_train['target']
#
Elo_train.describe(include='all')
print (Elo_train.describe(include='all'))
########## CREATION OF LINEAR REGRESSION
dates =Elo_train[Elo_train.columns[0]]#matplotlib.dates.date2num(list_of_datetimes)
Feat1 =Elo_train[Elo_train.columns[2]]
Feat0=Elo_test[Elo_test.columns[2]]
#print dates
X_train                   =Elo_train[Elo_train.columns[2:Elo_train_nbreRows[1]-1]]
X_test                    =Elo_test[Elo_test.columns[2:Elo_test_nbreRows[1]]]
Y_train                   =Elo_train_target
#####
print ('Test data isnull sum', X_test.isnull().sum())
####
print ('X-Train data NAN search', X_train.isnull().any())
print ('X-Test data NAN search',  X_test.isnull().any())
print ('Y-Train data NAN search', Y_train.isnull().any())
#### PLOT
plt.figure(1)
plt.scatter(Feat1,Y_train)
plt.xlabel("Y_Train")
plt.ylabel("Predicted Yi")
plt.title("Train vs Predicted ")
#########
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out
#X_trainC =remove_outlier(X_train,feature_3)
######
for i in range(1,2):
    lm   =LinearRegression() ### CREATE A LINEAR REGRESSION
    clf  =RandomForestRegressor(n_estimators=150*i, random_state=111)
    cltre=tree.DecisionTreeRegressor()
    model1 = KNeighborsRegressor(n_neighbors=i,weights='uniform')#,algorithm='brute')
    model2 = KNeighborsRegressor(n_neighbors=i,weights='distance')#,algorithm='brute')
    # ['auto', 'ball_tree', 'kd_tree', 'brute']
    ############
    X_train =pd.X_train
    X_train =pd.X_train
    X_train = X_train[X_train.between(X_train.quantile(.25), X_train.quantile(.75))] # without outliers
    Y_train = Y_train[Y_train.between(Y_train.quantile(.25), Y_train.quantile(.75))] # without outliers
    lm.fit(X_train,Y_train)
    clf.fit(X_train,Y_train)
    cltre.fit(X_train,Y_train)
    model1.fit(X_train,Y_train)
    model2.fit(X_train,Y_train)

    #print 29, clf.fit(X_train,Y_train)
    ##lm.fit(X, Dbos.PRICE)    ### FIT A LINEAR REGRESSION MODEL
    print ('Estimated intercept coefficient:', lm.intercept_)
    print ('Number of coefficients:', len(lm.coef_))
    ### CREATE A TABLE WHICH CONTAIN THE ENTIRE COEFFICIENTS 
    #Dcoef=pd.DataFrame(zip(X_train.columns,lm.coef_),columns=['features','Estimated Coefficients'])
    #print (Dcoef)
    ######## PREDICTED PRICES
    Y_pred1=lm.predict(X_test)
    Y_pred2=clf.predict(X_test)
    Y_pred3=cltre.predict(X_test)
##    Y_pred4=model1.predict(X_test)
##    Y_pred5=model2.predict(X_test)
    #print 500, len(Y_pred1),len(Y_pred2), len(Y_train)
    yy=list(range(0,len(Y_pred1)))
    xx=list(range(0,len(Y_train)))
    print (len(xx), len(yy))
    ###INTERPOLATION
    f0 =  np.interp(yy, xx,Y_train) 
####    f1 = np.interp(xx, yy,Y_pred1)               # Linear
####    f2 = np.interp(xx, yy,Y_pred2) # Cubic
####    f3= np.interp(xx, yy,Y_pred3) # Cubic
####    f4= np.interp(xx, yy,Y_pred4) # Cubic
####    f5= np.interp(xx, yy,Y_pred5) # Cubic
    print (len(Y_train),len(f0),len(Y_pred1),len(Y_pred2))#,len(f2)
    ######PLOT
    plt.figure(i+1)
    plt.scatter (Feat1,Y_train,c='b',s=40,alpha=0.5, label='Training data')
    plt.scatter (Feat0,Y_pred1,c='r',s=40,alpha=0.5, label='Predicted-Linear Regression')#,label1=EVS1)
    plt.scatter (Feat0,Y_pred2,c='g',s=40, alpha=0.5,label='Predicted-Random Forest')#
##    plt.scatter (Feat0,Y_pred3,c='m',s=49, alpha=0.5,label='Predicted-Decision Tree')#
    plt.xlabel("Feature_1")
    plt.ylabel("Y_train/Y_pred1/Y_pred2/Y_pred3")
    plt.title("Feature_1 vs prediction")
    plt.legend(loc='best')
    plt.show()
    ########## EVALUATION OF ACCURACY
    Score1 =np.sqrt(np.mean((f0-Y_pred1)**2) ) ##
    Score2 =np.sqrt(np.mean((f0-Y_pred2)**2) )
    Score3 =np.sqrt(np.mean((f0-Y_pred3)**2) )
    print (i, "LR RSME",format(Score1, '.3f'))
    print (i, "RF RSME",format(Score2, '.3f'))
    print (i, "DCT RSME",format(Score3, '.3f'))
    #####
 #Train_merged= pd.merge(Elo_train,Elo_newMerch, on='card_id', how='left')
    card_id= Elo_test['card_id']
    Elo_test['target']=Y_pred1
    target =Elo_test['target']
    df3= pd.concat([card_id,target], axis=1)
    print(df3.shape)
    print (df3.head(20))
DatatoSubmit=df3.to_csv('Submission.csv',index=False)
#print(DatatoSubmit.head(20))
##    print subtodata
