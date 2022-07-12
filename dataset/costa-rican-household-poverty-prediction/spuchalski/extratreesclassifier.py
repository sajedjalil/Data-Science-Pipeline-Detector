# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import pandas as pd
train= pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/train.csv', sep=',')
test= pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/test.csv', sep=',')

test["Target"]='dummy'

df=pd.concat([test,train])

numeric=[
'v2a1',
'rooms',
'v18q1',
'r4h1',
'r4h2',
'r4h3',
'r4m1',
'r4m2',
'r4m3',
'r4t1',
'r4t2',
'r4t3',
'tamhog',
'tamviv',
'escolari',
'rez_esc',
'hhsize',
'hogar_nin',
'hogar_adul',
'hogar_mayor',
'hogar_total', 
'dependency',
'edjefe',
'edjefa', 
'meaneduc',
'bedrooms', 
'overcrowding', 
'age',
'qmobilephone']

#derive categoric columns:
lista = df.columns.tolist()
new_lista= list()
for_removal = numeric+["Target",'idhogar','Id']
for item in lista:
    if item not in for_removal:
        new_lista.append(item)
categoric=new_lista[0:-9]
df[categoric].dtypes.unique()

#remove dependency and check for dtypes in the dataset
numeric.remove("dependency")
df[numeric].dtypes.unique()

df["edjefa"]=df["edjefa"].apply(lambda x: '0' if (x=="no" or x=="yes") else x).apply(lambda x: int(x))
df["edjefe"]=df["edjefe"].apply(lambda x: '0' if (x=="no" or x=="yes") else x).apply(lambda x: int(x))
df.loc[df["meaneduc"].isnull(),"meaneduc"] =df["meaneduc"].median()

#further clean the dataset:
df.loc[df["v18q1"].isnull(),"v18q1"]=0
df.loc[df["rez_esc"].isnull(),"rez_esc"]=0


#drop v2a1 since too many values are missing and combine the data into final form
data=df[numeric[1:]+categoric+["Target"]].copy()

test = data[data["Target"]=="dummy"].copy()
train = data[data["Target"]!="dummy"].copy()

#Mean encode
variables = train.iloc[:,:-1].copy()
dic={} #next an empty dictionary is created to store mean weights
#with the following loop for each variable I am deriving a dictionary of mean weights
#associated with the relationship of the target variable to each unique value in the column
for variable in variables.columns.tolist():
    means = {}
    for x in train[variable].unique():
        z = train[train[variable]== x]["Target"].mean()
        means[x]=z
    dic[variable]=means
    
    
train_variables = train.iloc[:,:-1].copy()
train_target = train.iloc[:,-1:].copy()

test_variables = test.iloc[:,:-1].copy()
test_target = test.iloc[:,-1:].copy()


def feature_mean(x,dic): 
    '''The function applies mean weights to each value in dataframe'''
    for key, value in dic.items():
        if x == key:
            return value

#In the following lines I am engineering new features by applying the created function

for key, value in dic.items(): 
    train_variables[key+"_feature_mean"] = train_variables[key].apply(lambda x: feature_mean(x,dic[key]))
    
for key, value in dic.items(): 
    test_variables[key+"_feature_mean"] = test_variables[key].apply(lambda x: feature_mean(x,dic[key]))
    
    
from sklearn.ensemble import ExtraTreesClassifier


#run the model on feature engineered dataset
mod = ExtraTreesClassifier(random_state=1, warm_start = False,bootstrap=False, max_features = 120, min_samples_leaf=1,
                                           min_samples_split=2,max_depth = 30, n_estimators=50)

mod.fit(train_variables,train_target.astype('int').values.ravel())
predicted = mod.predict(test_variables.fillna(test_variables.mean()))

test= pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/test.csv', sep=',')
id_test= test["Id"]

predicted = pd.Series(predicted, name='Target')

result = pd.concat([id_test,predicted], axis=1)
result.to_csv('Poverty_submission.csv', index=False)
