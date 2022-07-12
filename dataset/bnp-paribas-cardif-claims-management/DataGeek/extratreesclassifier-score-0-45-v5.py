import pandas as pd
import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB


def Binarize(columnName, df, features=None):
    df[columnName] = df[columnName].astype(str)
    if(features is None):
        features = np.unique(df[columnName].values)
    print(features)
    for x in features:
        df[columnName+'_' + x] = df[columnName].map(lambda y:
                                                    1 if y == x else 0)
    df.drop(columnName, inplace=True, axis=1)
    return df, features


def MungeData(train, test):

    features = train.columns[2:]
    print(type(features))
    for col in features:
        if((train[col].dtype == 'object') and (col!="v22")):
            print(col)
            train, binfeatures = Binarize(col, train)
            test, _ = Binarize(col, test, binfeatures)
            nb = BernoulliNB()
            nb.fit(train[col+'_'+binfeatures].values, train.target.values)
            train[col] = \
                nb.predict_proba(train[col+'_'+binfeatures].values)[:, 1]
            test[col] = \
                nb.predict_proba(test[col+'_'+binfeatures].values)[:, 1]
            train.drop(col+'_'+binfeatures, inplace=True, axis=1)
            test.drop(col+'_'+binfeatures, inplace=True, axis=1)
            train[col] = train[col].astype(float)
            test[col] = test[col].astype(float)
    return train, test


print('Load data...')
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()


#train, test = MungeData(train, test)
#train.head()





