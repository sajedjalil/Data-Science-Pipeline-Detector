#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 00:18:24 2017

@author: pegasus
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
dataset = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
dataset=dataset.iloc[:,1:]
# Write to the log:
pd.set_option('display.max_columns',None)
dataset.groupby("Cover_Type").size()
#print(dataset)
import numpy

size=10

data=dataset.iloc[:,:size]

cols=data.columns

data_corr=data.corr()

threshold=0.5
corr_list=[]

for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j])
            
s_corr_list=sorted(corr_list,key=lambda x: -abs(x[0]))

for v,i,j in s_corr_list:
    print("%s and %s = %.2f" % (cols[i], cols[j],v))

import seaborn as sns
import matplotlib.pyplot as plt
"""
for v,i,j in s_corr_list:
    sns.pairplot(dataset, hue="Cover_Type", size=6, x_vars=cols[i],y_vars=cols[j])
plt.show()

cols=dataset.columns

size=len(cols)-1

x=cols[size]
y=cols[0:size]

for i in range(0,size):
    sns.violinplot(data=dataset,x=x,y=y[i])
    plt.show()
    plt.savefig("output"+str(i)+".png")
# Any files you write to the current directory get shown as outputs

cols=dataset.columns

r,c=dataset.shape

data=pd.DataFrame(index=numpy.arange(0,r),columns=['Wilderness_Area','Soil_Type','Cover_Type'])

for i in range(0,r):
    w=0;
    s=0;
    for j in range(10,14):
        if(dataset.iloc[i,j]==1):
            w=j-9
            break
    for k in range(14,54):
        if(dataset.iloc[i,k]==1):
            s=k-13
            break
    data.iloc[i]=[w,s,dataset.iloc[i,c-1]]
    
sns.countplot(x="Wilderness_Area", hue="Cover_Type", data=data)
plt.show()
plt.savefig("output1.png")
plt.rc("figure", figsize=(25,10))
sns.countplot(x="Soil_Type", hue="Cover_Type", data=data)
plt.show()
plt.savefig("output2.png")
"""

rem=[]

for c in dataset.columns:
    if dataset[c].std()==0:
        rem.append(c)
        
dataset.drop(rem, axis=1,inplace=True)

print(rem)

r,c=dataset.shape

cols=dataset.columns

i_cols=[]
for i in range(0,c-1):
    i_cols.append(i)
ranks=[]

array=dataset.values

X=array[:,0:c-1]
Y=array[:,c-1]

val_size=0.1

seed=0
from sklearn import cross_validation

X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(X,Y,test_size=val_size, random_state=seed)
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

X_all=[]

X_all_add=[]

rem=[]
i_rem=[]

comb=[]
comb.append("All+1.0")

X_all.append(['Orig','All',X_train, X_val, 1.0, cols[:c-1],rem,ranks,i_cols,i_rem])
size=10

X_temp=StandardScaler().fit_transform(X_train[:,0:size])
X_val_temp=StandardScaler().fit_transform(X_val[:,0:size])

X_con=numpy.concatenate((X_temp,X_train[:,size:]),axis=1)
X_val_con=numpy.concatenate((X_val_temp, X_val[:,size:]),axis=1)

X_all.append(['StdSca','All',X_con,X_val_con,1.0,cols,rem,ranks,i_cols,i_rem])

X_temp=MinMaxScaler().fit_transform(X_train[:,0:size])
X_val_temp=MinMaxScaler().fit_transform(X_val[:,0:size])

X_con=numpy.concatenate((X_temp,X_train[:,size:]),axis=1)
X_val_con=numpy.concatenate((X_val_temp, X_val[:,size:]),axis=1)

X_all.append(['MinMax','All',X_con, X_val_con, 1.0, cols, rem, ranks, i_cols,i_rem])

X_temp=Normalizer().fit_transform(X_train[:,0:size])
X_val_temp=Normalizer().fit_transform(X_val[:,0:size])

X_con=numpy.concatenate((X_temp, X_train[:,size:]),axis=1)
X_val_con = numpy.concatenate((X_val_temp,X_val[:,size:]),axis=1)

X_all.append(['Norm','All',X_con, X_val_con, 1.0, cols, rem, ranks, i_cols,i_rem])

trans_list=[]

for trans, name, X, X_val, v, cols_list, rem_list, rank_list, i_cols_list, i_rem_list in X_all:
    trans_list.append(trans)

ratio_list=[0.75,0.50,0.25]

feat=[]

feat_list=[]

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

n='Extree'
feat_list.append(n)

for val in ratio_list:
    comb.append("%s+%s" % (n,val))
    feat.append([n,val,ExtraTreesClassifier(n_estimators=c-1,max_features=val,n_jobs=-1,random_state=seed)])
    
n='GraBst'
feat_list.append(n)
for val in ratio_list:
    comb.append("%s+%s" % (n,val))
    feat.append([n,val,GradientBoostingClassifier(n_estimators=c-1, max_features=val, random_state=seed)])
    
n='RndFst'
feat_list.append(n)
for val in ratio_list:
    comb.append("%s+%s" % (n,val))
    feat.append([n,val,RandomForestClassifier(n_estimators=c-1,max_features=val,n_jobs=-1,random_state=seed)])
    
n='XGB'
feat_list.append(n)
for val in ratio_list:
    comb.append("%s+%s" % (n,val))
    feat.append([n,val,XGBClassifier(n_estimators=c-1,random_state=seed)])

for trans,s,X,X_val,d,cols,rem,ra,i_cols,i_rem in X_all:
    for name,v,model in feat:
        model.fit(X,Y_train)
        joined=[]
        for i, pred in enumerate(list(model.feature_importances_)):
            joined.append([i,cols[i],pred])
        joined_sorted=sorted(joined,key=lambda x: -x[2])
        rem_start=int((v*(c-1)))
        cols_list=[]
        i_cols_list=[]
        rank_list=[]
        rem_list=[]
        i_rem_list=[]
        
        for j, (i,col,x) in enumerate(list(joined_sorted)):
            rank_list.append([i,j])
            if(j<rem_start):
                cols_list.append(col)
                i_cols_list.append(i)
            else:
                rem_list.append(col)
                i_rem_list.append(i)
                
        X_all_add.append([trans,name,X,X_val,v,cols_list,rem_list,[x[1] for x in sorted(rank_list,key=lambda x:x[0])],i_cols_list,i_rem_list])
        