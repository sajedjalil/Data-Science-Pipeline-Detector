# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')

print(train.head())
print(train.info())
#x=train.groupby('type')['color'].agg('count')
#print(x)
y_train=train['type']
x_train=train.drop(['color','id','type'],axis=1)
le = LabelEncoder().fit(y_train)
y_train = le.transform(y_train)
#le = LabelEncoder().fit(train['color'])
#y = le.transform(train['color'])

#sns.pointplot(x=x,y=y)  
#plt.show()
#colour is not significant in predicting
#so we drop colour
id=test['id']

print(x_train.head())
print(x_train.describe())
params = {'C':[1,5,10,0.1,0.01],'gamma':[0.001,0.01,0.05,0.5,1]}
log_reg = SVC()
#params={'min_samples_leaf':[40]}
clf = GridSearchCV(log_reg ,params, refit='True', n_jobs=1, cv=5)


clf.fit(x_train, y_train)
print(test.head())
x_test=test.drop(['id','color'],axis=1)
y_test = clf.predict(x_test)
print(clf.score(x_train,y_train))
#print(y_test[:])
y_test2=le.inverse_transform(y_test)
#print((clf.score(x_train,y_train)))
print('Best score: {}'.format(clf.best_score_))
print('Best parameters: {}'.format(clf.best_params_))
print(y_test)
submission = pd.DataFrame( { 
                  "type": y_test2
                   },index=id)
#submission.loc[submission["Y"]==1 , "Y"]=0
#submission.loc[submission["Y"]==-1 , "Y"]=0
submission.to_csv('submission3.csv')
