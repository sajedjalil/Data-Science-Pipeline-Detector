# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import ExtraTreesClassifier as Et 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

y=train['species']
#train=train.drop("species",1)
#print(train.head)

def encode(train, test):
    le = LabelEncoder().fit(train.species) 
    labels = le.transform(train.species)           # encode species strings
    classes = list(le.classes_)                    # save column names for submission
    test_ids = test.id                             # save test ids for submission
    
    train = train.drop(['species', 'id'], axis=1)  
    test = test.drop(['id'], axis=1)
    
    return train, labels, test, test_ids, classes

train, labels, test, test_ids, classes = encode(train, test)
train.head(1)

sss=StratifiedShuffleSplit(n_splits=2,random_state=42,test_size=0.25)

E=Et(verbose=1,n_estimators=400,random_state=42,n_jobs=-1)
print (E)

def Train_test(A,b):
    
    #X_train, X_test, y_train, y_test = train_test_split(A, b, test_size=0.33, random_state=42)
    train_index,test_index = sss.split(A, b)
    
    X_train, X_test = A.iloc[train_index[0]], A.iloc[test_index[1]]
    y_train, y_test = b.iloc[train_index[0]], b.iloc[test_index[1]]
    
    E.fit(X_train,y_train)
    print ("Extra T: ",E.score(X_test,y_test))
    #print("Log/loss",log_loss(X_test,y_test))
    
Train_test(train,y)
predict=E.predict_proba(test)


OP=pd.DataFrame(predict,columns=classes,index=test_ids)

#print(OP.head(10))

OP.to_csv("sub1.csv")










