
import pandas as pd

import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

print('Loading data...')
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#train = pd.read_csv("C://Users//SBT-Ashrapov-IR//Desktop//docs//apps//BNPParibasCardifClaimsManagement//train.csv")
#test = pd.read_csv("C://Users//SBT-Ashrapov-IR//Desktop//docs//apps//BNPParibasCardifClaimsManagement//test.csv")
target = train['target'].values
train = train.drop(['ID','target','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)
id_test = test['ID'].values
test = test.drop(['ID','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)


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
            train.loc[train_series.isnull(), train_name] = -999 
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -999

X_train = train
X_test = test


X_fit, X_eval, y_fit, y_eval= train_test_split(
    train, target, test_size=0.2, random_state=1
)

print('Training...')
clf = ExtraTreesClassifier(n_estimators=450,max_features= 60,criterion= 'entropy',min_samples_split= 4,
                            max_depth= 25, min_samples_leaf= 2)      

clf.fit(X_fit,y_fit ) 

print('Predicting...')
y_pred = clf.predict_proba(X_test)
#print y_pred

pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:,1]}).to_csv('extra_trees.csv',index=False)

auc_train = mean_squared_error(y_fit, clf.predict_proba(X_fit)[:,1])
auc_valid = mean_squared_error(y_eval, clf.predict_proba(X_eval)[:,1])

print('\n-----------------------')
print('  AUC train: %.5f'%auc_train)
print('  AUC valid: %.5f'%auc_valid)
print('-----------------------')

print('\nModel parameters...')
print('\n-----------------------\n')