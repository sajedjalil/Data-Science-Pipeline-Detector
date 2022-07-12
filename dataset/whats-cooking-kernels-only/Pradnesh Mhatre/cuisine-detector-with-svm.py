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


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pandas as pd
import json

# Dataset Preparation
print ("Read Dataset ... ")
def read_dataset(path):
    return json.load(open(path)) 
train = read_dataset('/kaggle/input/whats-cooking-kernels-only/train.json')
test = read_dataset('/kaggle/input/whats-cooking-kernels-only/test.json')

# Text Data Features
print ("Prepare text data of Train and Test ... ")

data_list_tr = []
data_list_test = []

for item in train:
    data_dict = {"cuisine":None, "ingredients":None}
    data_dict['cuisine'] = item['cuisine']
    data_dict['ingredients'] = item['ingredients']
    data_list_tr.append(data_dict)

for item in test:
    data_dict = {"id":None, "ingredients":None}
    data_dict['id'] = item['id']
    data_dict['ingredients'] = item['ingredients']
    data_list_test.append(data_dict)

df_tr=pd.DataFrame(data_list_tr, columns=['ingredients', 'cuisine'])
df_test=pd.DataFrame(data_list_test, columns=['id', 'ingredients'])
df_tr['ingredients']=df_tr['ingredients'].astype(str)
df_test['ingredients']=df_test['ingredients'].astype(str)

df_tr['ingredients']=df_tr['ingredients'].str.replace("[","")
df_tr['ingredients']=df_tr['ingredients'].str.replace("]","")
df_tr['ingredients']=df_tr['ingredients'].str.replace("'","")
df_tr['ingredients']=df_tr['ingredients'].str.replace(",","")
df_tr['ingredients']=df_tr['ingredients'].str.replace("-"," ")

df_test['ingredients']=df_test['ingredients'].str.replace("[","")
df_test['ingredients']=df_test['ingredients'].str.replace("]","")
df_test['ingredients']=df_test['ingredients'].str.replace("'","")
df_test['ingredients']=df_test['ingredients'].str.replace(",","")
df_test['ingredients']=df_test['ingredients'].str.replace("-"," ")

from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_vect = TfidfVectorizer() 
tf_idf_vect.fit_transform(df_tr['ingredients'].values)
tfidf_train=tf_idf_vect.fit_transform(df_tr['ingredients'].values)
tfidf_test=tf_idf_vect.transform(df_test['ingredients'])

from sklearn.svm import SVC
svm_model_rbf = SVC(kernel ='rbf', C = 1000,degree=1,
          gamma=1, coef0=1, shrinking=True, 
          probability=False, tol=0.001, cache_size=200,
          class_weight=None, verbose=True, max_iter=-1,
          decision_function_shape='ovr', random_state=None)
clf=svm_model_rbf.fit(tfidf_train, df_tr['cuisine'])


df_test['cuisine'] = clf.predict(tfidf_test)
df1_test=df_test
df1_test=df1_test.drop('ingredients',axis=1)

df1_test.to_csv("submission.csv", index = False)