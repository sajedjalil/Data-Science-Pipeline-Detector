# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import sklearn
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import  SVC,LinearSVC
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,log_loss,jaccard_similarity_score
from sklearn.preprocessing import normalize
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
# Dataframe Read
df_train =pd.read_csv("../input/train.csv")
df_test=pd.read_csv("../input/test.csv")

# Analysing Target Dataset
df_train.head()
df_train["Cover_Type"].describe()
df_Target=df_train["Cover_Type"]
df_train[["Elevation","Cover_Type"]].groupby("Cover_Type").hist()
df_plot=df_train[['Elevation','Cover_Type']]
scatter_matrix(df_plot, alpha=0.2, figsize=(6, 6), diagonal='kde')

df_train[["Elevation"]].hist()
df_test[["Elevation"]].hist()

# Range of elevation  is different so lets drop it
df_train=df_train.drop('Elevation',axis=1)
df_test=df_test.drop('Elevation',axis=1)
# Train/Test Data analyse
#Lets normalize dataset
colstoNormalize=[]
for i in df_train:
    if i != "Cover_Type" and i != "Id":
        if max(df_train[i])>1:
            print(max(df_train[i]))
            colstoNormalize.append(i)
df_train[colstoNormalize]=normalize(df_train[colstoNormalize])
df_test[colstoNormalize]=normalize(df_test[colstoNormalize])
# checking data is really Normalized or not
for i in df_train:
    if i != "Cover_Type" and i != "Id":
        print(max(df_train[i]))

# Lets drop ID{ Not usefull} and split the class,target
df_train=df_train.drop(columns=["Id"])
df_test_id=df_test['Id']
df_test=df_test.drop(columns=["Id"])
# Now we convert dataframe to numpy array
df_Y_train=df_train["Cover_Type"]
df_X_train=df_train.drop("Cover_Type",axis=1)
## Lets Split dataset in train and test

X_train,X_test,Y_train,Y_test=train_test_split(df_X_train,df_Y_train,test_size=0.20,random_state=42)
#Now we are ready for training our classifiers
#We will start with random forest classifier. I will be using Default parameters for train and later we will be optimizing parameters
# from sklearn.datasets import make_classification
# X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)

#first initilize a random froest classifier
rf1 = RandomForestClassifier(n_estimators=144, max_depth=8,random_state=0)
rf1.fit(X_train,Y_train)
y_pred=rf1.predict(X_test) #thats it We are ready with initial classification. Lets evalute results How accurate our classifier is
accuracy_score(Y_test,y_pred)
#our individual classfier's test accuracy is around 70% to be percise in my case it was 70.12%
#We can furter improve this accuracy by optimizing RF parameters. That will be an other time discussion topic, Our objective is to improve accuracy by stacking multiple classifier
#now lets train Multiple random forest by varying Random forest parameters. I will select intiger parameters in which are part of fibonacci series.

#First 10 elements of fibonacci series are as following
# 0	1	1	"2	3	5	8	13	21	34	55	89	144	233	377	610	987	1597	2584	4181	6765 "

#RF 2
rf2 = RandomForestClassifier(n_estimators=233, max_depth=13,random_state=0)
rf2.fit(X_train,Y_train)
y_pred=rf2.predict(X_test) #thats it We are ready with initial classification. Lets evalute results How accurate our classifier is
accuracy_score(Y_test,y_pred)

#RF 3
rf3 = RandomForestClassifier(n_estimators=377, max_depth=13,random_state=0)
rf3.fit(X_train,Y_train)
y_pred=rf3.predict(X_test) #thats it We are ready with initial classification. Lets evalute results How accurate our classifier is
accuracy_score(Y_test,y_pred)

#RF 4

rf4 = RandomForestClassifier(n_estimators=377, max_depth=21,random_state=0,criterion="entropy")
rf4.fit(X_train,Y_train)
y_pred=rf4.predict(X_test) #thats it We are ready with initial classification. Lets evalute results How accurate our classifier is
accuracy_score(Y_test,y_pred)

# For stacking we will be using heamy library


from heamy.pipeline import ModelsPipeline
from heamy.estimator import Classifier

dataset={'X_train':X_train.values, 'y_train':Y_train.ravel()-1, 'X_test':X_test.values, 'y_test':Y_test.ravel()-1}


from sklearn.ensemble import VotingClassifier



if 0:
    parameters = {'n_estimators': 144, 'max_depth': 8, 'random_state': 0}
    rf1 = Classifier(dataset=dataset, estimator=RandomForestClassifier, parameters=parameters, name='rf1')
    ##RF2
    parameters = {'n_estimators': 233, 'max_depth': 13, 'random_state': 0}
    rf2 = Classifier(dataset=dataset, estimator=RandomForestClassifier, parameters=parameters, name='rf2')
    ##RF2
    parameters = {'n_estimators': 377, 'max_depth': 13, 'random_state': 0}
    rf3 = Classifier(dataset=dataset, estimator=RandomForestClassifier, parameters=parameters, name='rf3')
    ##RF2
    parameters = {'n_estimators': 377, 'max_depth': 21, 'random_state': 0}
    rf4 = Classifier(dataset=dataset, estimator=RandomForestClassifier, parameters=parameters, name='rf4')
    #some extratree classfier
    #ET1
    parameters={'n_estimators':377, 'max_depth':21,'random_state':0}
    et1=Classifier(dataset=dataset,estimator=ExtraTreesClassifier,parameters=parameters,name='et1')
    #ET2
    parameters={'n_estimators':377, 'max_depth':8,'random_state':0}
    et2=Classifier(dataset=dataset,estimator=ExtraTreesClassifier,parameters=parameters,name='et2')

    #LGBM Classsifier
    parameters={'n_estimators':377, 'learning_rate':0.1}
    lgbm1=Classifier(dataset=dataset,estimator=LGBMClassifier,parameters=parameters,name='lgbm1')

    #Gradient boosting classifier
    parameters={'n_estimators':377, 'learning_rate':0.1}
    gb1=Classifier(dataset=dataset,estimator=GradientBoostingClassifier,parameters=parameters,name='gb1')

    # Adaboost classifier
    parameters={'n_estimators':377, 'learning_rate':0.1,'random_state':0}
    adboost1=Classifier(dataset=dataset,estimator=AdaBoostClassifier,parameters=parameters,name='adboost1')

    #Logistic regrission
    parameters={'solver':'liblinear','multiclass':'ovr','random_state':0}
    logisticR1=Classifier(dataset=dataset,estimator=LogisticRegression,parameters=parameters,name='logisticR1')

    #XGBoost
    parameters={'solver':'liblinear','num_class':7,'learning_rate':0.1,'random_state':0}
    XGB1=Classifier(dataset=dataset,estimator=XGBClassifier,parameters=parameters,name='XGB1')
else:
    rf1 = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0)
    rf2 = RandomForestClassifier(n_estimators=233, max_depth=13, random_state=0)
    rf3 = RandomForestClassifier(n_estimators=377, max_depth=30, random_state=0)
    rf4 = RandomForestClassifier(n_estimators=377, max_depth=21, random_state=0, criterion="entropy")
    et1=ExtraTreesClassifier(n_estimators=377, max_depth=21,random_state=0)
    et2 = ExtraTreesClassifier(n_estimators=377, max_depth=8, random_state=0)
    lgbm1=LGBMClassifier(n_estimators=377,learning_rate=0.1)
    gb1=GradientBoostingClassifier(n_estimators=377,learning_rate=0.1)
    adboost1=AdaBoostClassifier(n_estimators=377,learning_rate=0.1,random_state=0)
    logisticR1=LogisticRegression(solver='liblinear',multi_class='ovr',random_state=0)
    XGB1=XGBClassifier(n_estimators=200,objective='gbtree')

model = VotingClassifier(estimators=[('rf1', rf1), ('rf2', rf2),('rf3',rf3),('rf4',rf4),('et1',et1),('et2',et2),('lgbm1',lgbm1),('gb1',gb1),('adboost1',adboost1),('logisticR1',logisticR1),('XGB1',XGB1)], voting='hard',n_jobs=10)
# model = VotingClassifier(estimators=[('rf1', rf1), ('rf2', rf2),('rf3',rf3),('rf4',rf4)], voting='hard')
model.fit(X_train,Y_train)
#
# pipeline =ModelsPipeline(rf1,rf2,rf3,rf4,et1,et2,lgbm1,gb1,adboost1,logisticR1,XGB1)
# stack_ds=pipeline.stack(k=4)
# 64from sklearn.pipeline import Pipeline
# anova_svm = Pipeline([('rf1', rf1), ('rf2', rf2)])
model.score(X_test,Y_test)
print("Predictiong")
Predicted=model.predict(df_test)
dfTest=pd.DataFrame({'Id':df_test_id,'Cover_Type':Predicted})
dfTest.to_csv('sample_submission.csv',index=False,encoding='utf-8')