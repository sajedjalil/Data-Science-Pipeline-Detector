


from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble  import RandomForestClassifier
from sklearn import neighbors, datasets
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
from sklearn.tree     import DecisionTreeClassifier
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold

from sklearn.linear_model import LogisticRegression


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


X,y,dict=getTrainData(size=None)
print("input loaded!")

clfs=\
    [
    #   LogisticRegression(solver='newton-cg')
        #svm.SVC(),
    # NearestNeighbors(n_neighbors=7, algorithm='ball_tree'),
     #neighbors.KNeighborsClassifier(15,weights='distance'),
     RandomForestClassifier(max_features=50, n_estimators=20),
     #RandomForestClassifier(n_estimators=2000)
     #neighbors.KNeighborsRegressor(7, weights='distance')
     ]
     
scoreList=[]

for clf in clfs:
    print(clf)
    for i in range(5):
        clf.fit(X,y)
        folding=KFold(X.shape[0],10,shuffle=True)
        scores = cross_val_score(clf, X, y,cv=folding)#,scoring="accuracy"
        scoreList.append(scores.mean())
        print(scoreList)
