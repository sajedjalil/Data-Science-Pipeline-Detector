import pandas as pd
import numpy as np
import scipy as sc
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

###########  Data Loading ##################################################
train_df = pd.read_csv('../input/train.csv', encoding="ISO-8859-1", header=0)
test_df = pd.read_csv('../input/test.csv',  encoding="ISO-8859-1", header=0)
print ("loading done")


########## Data cleaning ##################################################
train_y = train_df['place_id']
train_df.drop(['row_id'],axis=1,inplace=True)
train_df.drop(['accuracy'],axis=1,inplace=True)
train_df.drop(['place_id'],axis=1,inplace=True)

row_id = test_df['row_id']
test_df.drop(['row_id'],axis=1,inplace=True)
test_df.drop(['accuracy'],axis=1,inplace=True)

print ("cleaning done")

train_X = train_df.as_matrix()
test_X = test_df.as_matrix()


###############     Learning and prediction        ############################################

knn = KNeighborsClassifier(n_neighbors=10)
#clf = OneVsRestClassifier(knn)
#clf.multilabel_ = True
#clf.fit(train_X,train_y)
knn.fit(train_X,train_y)
predictions = knn.predict(test_X)


_submit = open("submit.csv",'w')
_submit.write("row_id,place_id\n")
for i in range(0,len(predictions)):
        _submit.write(str(row_id[i])+",")
        _submit.write(" "+str(predictions[i])+"\n")

_submit.flush()
_submit.close()
