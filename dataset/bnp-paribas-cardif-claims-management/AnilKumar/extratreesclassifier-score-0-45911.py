import pandas as pd
import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble


print('Load data...')
train = pd.read_csv("../input/train.csv")
target = train['target'].values
train = train.drop(['ID','target'],axis=1)
test = pd.read_csv("../input/test.csv")
id_test = test['ID'].values
test = test.drop(['ID'],axis=1)

print('Clearing...')
for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_series.dtype == 'O':
        #for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
        #but now we have -1 values (NaN)
    else:
        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            #print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = -9999 #train_series.mean()
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -9999 #train_series.mean()  #TODO

X_train = train
X_test = test

extc = ExtraTreesClassifier(n_estimators=700,max_features= 50,criterion= 'entropy',min_samples_split= 4,
                            max_depth= 50, min_samples_leaf= 4)      

extc.fit(X_train,target) 

print('Predict...')
y_pred = extc.predict_proba(X_test)
#print y_pred

pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:,1]}).to_csv('extra_trees.csv',index=False)