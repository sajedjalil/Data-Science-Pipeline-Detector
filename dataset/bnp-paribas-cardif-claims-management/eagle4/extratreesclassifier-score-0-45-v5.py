import pandas as pd






#########################
### use https://www.kaggle.com/chabir/bnp-paribas-cardif-claims-management/extratreesclassifier-score-0-45-v5/run/178143
### to get about .4548
#################








import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble


print('Load data...')
train = pd.read_csv("../input/train.csv")
target = train['target'].values
train = train[['v3','v10','v12','v14','v21','v22','v24','v30','v31','v34','v38','v40','v47','v50','v52','v56','v62','v66','v71','v72','v74','v75','v79','v91','v107','v110','v112','v113','v114','v125','v129']]
test = pd.read_csv("../input/test.csv")
id_test = test['ID'].values
test = test[['v3','v10','v12','v14','v21','v22','v24','v30','v31','v34','v38','v40','v47','v50','v52','v56','v62','v66','v71','v72','v74','v75','v79','v91','v107','v110','v112','v113','v114','v125','v129']]




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
print('Training...')
extc = ExtraTreesClassifier(n_estimators=1200,max_features= 30,criterion= 'entropy',min_samples_split= 2,
                            max_depth= 30, min_samples_leaf= 2, n_jobs = -1)    

extc.fit(X_train,target) 

print('Predict...')
y_pred = extc.predict_proba(X_test)
#print y_pred

pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:,1]}).to_csv('extra_trees.csv',index=False)