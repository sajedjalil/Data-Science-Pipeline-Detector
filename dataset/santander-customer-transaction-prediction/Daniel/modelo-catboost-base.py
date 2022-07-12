# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import Pool, CatBoostClassifier, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

X = train_df.drop(['ID_code','target'], axis=1) # Features
y = train_df.target.values # Target variable


X_test_pred = test_df.drop(['ID_code'], axis=1)

train_df.info()

train_df.isnull().sum()

#Dividimos X e y en datos de entrenamiento y prueba
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#Se crea el modelo
model=CatBoostClassifier(eval_metric="AUC", iterations=40, depth=5, learning_rate=0.15, use_best_model=True)

#Se entrena el modelo y se previene el sobreajuste
model.fit(X_train,y_train,eval_set=(X_test,y_test))

y_pred_test = model.predict(X_test_pred)
#creacion del archivo
submission = pd.DataFrame({'ID_code':test_df.ID_code,'target':y_pred_test})
submission.to_csv('submission.csv', index=False)