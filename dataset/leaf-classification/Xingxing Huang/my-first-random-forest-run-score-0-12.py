import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV as cc
import pdb
from pylab import * 
ion()


from subprocess import check_output
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss,roc_auc_score
from sklearn.grid_search import GridSearchCV


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test_ids = test.id 

levels=train.species
train.drop(['species', 'id'], axis=1,inplace=True) 
test.drop(['id'],axis=1,inplace=True)
columns = train.columns


le=LabelEncoder().fit(levels)
levels=le.transform(levels)
lb = LabelBinarizer().fit(levels)


## rescale 
ss = StandardScaler().fit(train)
#ss = MinMaxScaler().fit(train)
train = ss.transform(train)
test = ss.transform(test)
train = pd.DataFrame(train, columns = columns)
test = pd.DataFrame(test, columns = columns)

# shuffle 
sss = StratifiedShuffleSplit(levels, 1, test_size=0.2, random_state=11)  # split to 10 training
for train_index, test_index in sss:
    print ('\n New..')
    X_train, X_test = train.values[train_index], train.values[test_index]
    y_train, y_test = levels[train_index], levels[test_index]
    
    
    # alg_frst_params = [{
    #     "n_estimators": [10, 100, 500, 1000],
    #     "min_samples_split": [2,  8,  20, 100],
    #     "min_samples_leaf": [1, 2,3,4,5,6,7, 8,9]
    # }]
    
    # alg_frst_params = [{
    #     "n_estimators": [10, 100, 500],
    #     "min_samples_split": [2,  8, ],
    #     "min_samples_leaf": [2,  8]
    # }]
    # alg_frst_model=RandomForestClassifier()
    # favorite_clf = GridSearchCV(alg_frst_model, alg_frst_params, scoring = 'accuracy',  refit=True, verbose=1, n_jobs=-1)
    # favorite_clf.fit(X_train, y_train)
    
    
    Model=RandomForestClassifier(n_estimators=1000,min_samples_split=2,min_samples_leaf=2,oob_score=True,random_state=42 )
    #Model=RandomForestClassifier(n_estimators=100,oob_score=True,random_state=42 )
    Model = cc(Model, cv=3, method='isotonic')
    Model.fit(train, levels)

    
    predictions_prob = Model.predict_proba(X_train)
    predictions_test_prob = Model.predict_proba(X_test)
    predictions = Model.predict(X_train)
    predictions_test = Model.predict(X_test)
    
    #print Model.oob_score_
    # multi class not supported
    #pdb.set_trace()
    #print 'c-state', roc_auc_score(y_train,predictions)
    #print 'c-state', roc_auc_score(y_train,predictions_prob)
    
    # the accurate
    acc = accuracy_score(y_train, predictions)
    acc_test = accuracy_score(y_test, predictions_test)
    print("Accuracy: %10.4f  %10.4f"  %(acc, acc_test))

    # the log cos 
    predictions_binarizer = lb.transform(predictions)
    predictions_test_binarizer = lb.transform(predictions_test)

    ll_train = log_loss(y_train, predictions_prob)
    ll_test = log_loss(y_test, predictions_test_prob)
    print("Log Loss: %10.4f  %10.4f"  %(ll_train, ll_test))
    ll_train = log_loss(y_train, predictions_binarizer)
    ll_test = log_loss(y_test, predictions_test_binarizer)
    print("Log Loss: %10.4f  %10.4f"  %(ll_train, ll_test))
    
# pdb.set_trace()
####

Model=RandomForestClassifier(n_estimators=1000,oob_score=True )
Model = cc(Model, cv=3, method='isotonic')
Model.fit(train, levels)
predictions = Model.predict(test)
predictions_proba = Model.predict_proba(test)
predictions_binarizer = lb.transform(predictions)



# sub = pd.DataFrame(predictions_proba, columns=list(le.classes_))
# sub.insert(0, 'id', test_ids)
# sub.reset_index()
# sub.to_csv('submit.csv', index = False)
# sub.head(2)    
# pdb.set_trace()


sub = pd.DataFrame(predictions_proba, columns=list(le.classes_))
#func = lambda x: x  if (x<0.6) and (x>0.4) else round(x)
#sub_new = sub.applymap(func)
sub_new = sub
sub_new.insert(0, 'id', test_ids)
sub_new.reset_index()
sub_new.to_csv('submit.csv', index = False)
sub_new.head(2)    

#pdb.set_trace()