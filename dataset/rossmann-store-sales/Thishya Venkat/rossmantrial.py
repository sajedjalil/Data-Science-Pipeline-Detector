import pandas as pd

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print(train.describe())

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 00:48:32 2015

@author: vaibhavgedigeri
"""

import pandas as pd

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print(train.describe())

# coding: utf-8

# ## DATA IMPORT

# In[1]:

# imports
import pandas as pd
import numpy as np
import sklearn.externals.joblib as jl

import sklearn.cross_validation as cv
import sklearn.feature_extraction as fe

import sklearn.svm as svm

import matplotlib.pyplot as plt

import sklearn.linear_model as lm

import sklearn.preprocessing as preprocessing

import sklearn.ensemble as es


# In[2]:

# import datas

# Three original frame : store ,  train, test


store = pd.read_csv('../input/store.csv')
train = pd.read_csv('../input/train.csv',low_memory=False)
test = pd.read_csv('../input/test.csv')

print("Reading CSV Done.")

# In[3]:

df_train = train.copy()
df_test = test.copy()

print("Assume store open, if not provided")
df_test.fillna(1, inplace=True)

print("Consider only open stores for training. Closed stores wont count into the score.")
df_train = df_train[df_train["Open"] != 0]

# In[21]:


df_train = pd.merge(train,store,on='Store')
df_test = pd.merge(test,store,on='Store')


# In[5]:

#df_train = df_train.sample(100000)




# In[22]:


sale_means = train.groupby('Store').mean().Sales
sale_means.name = 'Sales_Means'

df_train = df_train.join(sale_means,on='Store')
df_test = df_test.join(sale_means,on='Store')




# ## Transform dataframe to Matrix

# In[23]:

y = df_train.Sales.tolist()

df_train_ = df_train.drop(['Date','Sales','Store','Customers'],axis=1).fillna(0)



train_dic = df_train_.fillna(0).to_dict('records')


test_dic = df_test.drop(["Date","Store","Id"],axis=1).fillna(0).to_dict('records')



# In[24]:

#transfrom dataframe to matrix by dict vectorizer
dv = fe.DictVectorizer()
X = dv.fit_transform(train_dic)
Xo = dv.transform(test_dic)


# In[25]:

#MIN_MAX SCALER
maxmin = preprocessing.MinMaxScaler()
X = maxmin.fit_transform(X.toarray())
Xo = maxmin.transform(Xo.toarray())


# In[26]:

Xtrain,Xtest,Ytrain,Ytest = cv.train_test_split(X,y)



# ###### MODEL SELECTION

# In[27]:

clf = es.RandomForestRegressor(n_estimators=25)
clf.verbose = True
clf.n_jobs = 8
clf


# In[28]:

clf.fit(Xtrain,Ytrain)
print ("Training Score :" + str(clf.score(Xtrain,Ytrain)))
print ("Test Score : " + str(clf.score(Xtest,Ytest)) )


# In[34]:

q = [i for i in zip(dv.feature_names_,clf.feature_importances_) ]

q = pd.DataFrame(q,columns = ['Feature_Names','Importance'],index=dv.feature_names_)

q_chart = q.sort('Importance').plot(kind='barh',layout='Feature_Names')

fig_q = q_chart.get_figure()
fig_q.savefig('feature_impartance.png')


# In[35]:

Yresult = clf.predict(Xtest)
Yresult = np.array(Yresult)

Ytest = np.array(Ytest)

np.abs((Yresult - Ytest)).sum() / len(Yresult)


# ##### PREDICTION

# In[36]:


result = clf.predict(Xo)


# In[37]:

output = pd.DataFrame(df_test.Id).join(pd.DataFrame(result,columns=['Sales']))


# In[38]:

output.to_csv('outputx.csv',index=False)