import pandas as pe
import numpy as ny
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn import svm
import csv

#importing train data
train = pe.read_csv('../input/train.csv')
#importing test data
test = pe.read_csv('../input/test.csv')
#data preprocessing
dl=LabelEncoder().fit(train.species)
labels = dl.transform(train.species)
classs = list(dl.classes_)
train=train.drop(["species","id"],axis=1)
#to create test data for submission
testid=test.id
test=test.drop(["id"],axis=1)
#random number for model to train again and again
ny.random.uniform(5)


sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)

for train_index, test_index in sss:
    X_train, X_test = train.values[train_index], train.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

clf = svm.SVC(kernel="rbf", C=1000, degree=5, probability=True, max_iter=1000,tol=0.1, gamma=3)
clf.fit(X_train,y_train)
train_predictions = clf.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
test_predictions = clf.predict_proba(test)
print(acc)
submission = pe.DataFrame(test_predictions, columns=classs)
#print classs
submission.insert(0, 'id', testid)
submission.to_csv('submission1.csv',index_label=None,index=False)
'''
a=type(submission)
b=pe.read_csv('submission.csv')
c=b.insert(0, )
b1=b.drop(["pop"],axis=1)'''
#submission.to_csv('submission2.csv',index_label=None)

#print a
