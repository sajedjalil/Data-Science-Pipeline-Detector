
import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing

#from sklearn.cross_validation import train_test_split



# gia tin aksiologisi twn classifiers xrisimopoioume auta gia training k testing anti ta katw
'''
g = pd.read_csv("../input/train.csv")
train, test, y, y_test = train_test_split(g, g['target'], test_size=0.3, random_state=42)
y = train['target'].values
trainID = train['ID'].values
train = train.drop(['ID','target'],axis=1)
testID = test['ID'].values
test = test.drop(['ID','target'],axis=1)
'''

train = pd.read_csv("../input/train.csv")
y = train['target'].values
trainID = train['ID'].values
train = train.drop(['ID','target'],axis=1)


test = pd.read_csv("../input/test.csv")
testID = test['ID'].values
test = test.drop(['ID'],axis=1)

#for each feature in training set
for i in train:
    if train[i].dtype == 'O':
        #http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
        train[i] = np.array(train[i],dtype=str)
        train[i] = preprocessing.LabelEncoder().fit_transform(train[i])
        
    else:
        #fill null value with mean value of column (fetaure)
        tmp_len = len(train[train[i].isnull()])
        if tmp_len>0:
            #train[i] = train[i].fillna(float(train[i].mean()))
            train.loc[train[i].isnull()] = train[i].mean()
   
   
#GaussianNB

'''
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(train, y)

'''


# Logistic Regression

'''
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(train, y)
'''

#ExtraTreesClassifier

#http://scikit-learn.org/stable/modules/ensemble.html#ensemble
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=1000,max_features= 'sqrt',criterion= 'entropy', n_jobs = -1 )
clf.fit(train, y)



print('End of the training...')




for i in test:
    if test[i].dtype == 'O':
        
        test[i] = np.array(test[i],dtype=str)
        test[i] = preprocessing.LabelEncoder().fit_transform(test[i])
    
    else:
        tmp_len = len(test[test[i].isnull()])
        if tmp_len>0:
            #test[i] = test[i].fillna(float(test[i].mean()))
            test.loc[test[i].isnull()] = test[i].mean()
            

#probability 
PredictedProb = clf.predict_proba(test)
 
print(PredictedProb)
pd.DataFrame({"ID": testID, "PredictedProb": PredictedProb[:,1]}).to_csv('predicted_test_params.csv',index=False)

# gia tin aksiologisi twn classifiers
'''
Predictions = clf.predict(test)

from sklearn.metrics import log_loss
print(log_loss(y_test,PredictedProb))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,Predictions))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, Predictions, labels=[1,0]))

'''


