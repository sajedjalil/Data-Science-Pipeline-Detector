import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm


def categoricalToNumerical(v,dict=None):
    if not dict:
        dict={}
    biggest=len(dict)
    res=[]
    for i in v:
        if not i in dict:
            dict[i]=biggest
            biggest+=1
        res.append(dict[i])
    return np.array(res),biggest,dict


def getTrainData(size=None,featureIndices=None):
    train = pd.read_csv('../input/train.csv')
    train=train.fillna(0)
    # train=train.fillna(train.mean())
    train=train.values
    if size:
        train=train[:size,:]
    if featureIndices:
        indices=featureIndices
    else:
        indices=list(range(1,train.shape[1]-1))#All indices except ID and except response
    # print(indices)
    changed=categoricalToNumerical(train[:,2])
    dict=changed[2]
    print("number of categories of column 2 of train:",changed[1])
    train[:,2]=changed[0]

    X, y = train[:,indices] , train[:,-1]

    y=list(map(np.int32,y))
    return X,y,dict


def getTestData(dict,size=None,featureIndices=None,):
    test = pd.read_csv('../input/test.csv')
    test=test.fillna(0)
    # train=train.fillna(train.mean())
    test=test.values
    if size:
        test=test[:size,:]
    if featureIndices:
        indices=featureIndices
    else:
        indices=list(range(1,test.shape[1]))#All indices except ID and except response
    # print(indices)
    changed=categoricalToNumerical(test[:,2],dict)
    print("number of categories of column 2 of train and test:",changed[1])
    test[:,2]=changed[0]

    X= test[:,indices]
    ids=test[:,0]
    return X,ids


def saveToFile(predictions,ids,filename):

    submission = pd.DataFrame({
    "Id": ids,
    "Response": predictions
    })
    submission.to_csv(filename, index=False)



# importance=[10,9,3,7,40,8,11,16,1,38,37,19,35,33,12,59,34,51,
# 92,28,36,32,77,22]

size=None

Test = pd.read_csv('../input/test.csv')
X,y,dict=getTrainData(size=size)
Test,ids=getTestData(dict,size=size)

print("Data loaded!")
# X=X[:,importance]
# Test=Test[:,importance]
clf = RandomForestClassifier(max_features=70, n_estimators=500)
clf=clf.fit(X,y)
ytest=clf.predict(Test)
print(ytest.shape)
saveToFile(ytest,ids,"testSubmission.csv")

