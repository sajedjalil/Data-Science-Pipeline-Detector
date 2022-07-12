# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neural_network import MLPClassifier as M
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

y=train['species']

le = LabelEncoder().fit(train.species) 
labels = le.transform(train.species)           # encode species strings
classes = list(le.classes_)                    # save column names for submission
test_ids = test.id                             # save test ids for submission
    
train = train.drop(['species', 'id'], axis=1)  
test = test.drop(['id'], axis=1)
    
MLP=M(hidden_layer_sizes=(192,512,128),random_state=42,solver='adam',tol=0.00001,activation='tanh',learning_rate='adaptive',verbose=10)
Scale=StandardScaler()

T=Scale.fit_transform(train)

MLP.fit(T,labels)

predict=MLP.predict_proba(Scale.transform(test))

OP=pd.DataFrame(predict,columns=classes,index=test_ids)

#print(OP.head(10))

OP.to_csv("MLP.csv")