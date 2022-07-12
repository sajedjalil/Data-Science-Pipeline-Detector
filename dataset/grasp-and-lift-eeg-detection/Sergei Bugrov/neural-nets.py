#!/usr/bin/python

import numpy as np
from sklearn import cross_validation

from sklearn import svm

import pandas as pd
import math as m
from sklearn import preprocessing

from sklearn.metrics import mean_squared_error as MSE


subjects = range(1, 2) #13)
train_series = range(1, 2) # 9
test_series = range(9, 10) # 11

for subject in subjects:
	fname = ('../input/train/subj%d_series1_data.csv' % (subject))
	train = pd.read_csv(fname)
	fname = ('../input/train/subj%d_series1_events.csv' % (subject))
	labels = pd.read_csv(fname)
	
	train = pd.merge(train, labels, on ='id')
	
	idx = train.id.values

	train = train.drop(['id'], axis = 1)
	columns = train.columns[ :-6 ]

	train = np.array(train).astype(float)
	labels = np.array(labels)
	
	x_train = train[ :, :-6 ]
	y_train = train[ :, 32: ]

	colcount = x_train.shape[1]
	rowcount = x_train.shape[0]

	for i in range(0, colcount):
		x_train[:, i] = preprocessing.scale(x_train[:, i])
		mean = np.mean(x_train[:, i])
		max = np.max(x_train[:, i])
		min = np.min(x_train[:, i])
		for j in range(0, rowcount):
			x_train[j, i] = ( x_train[j, i] - min )/( max - min )
				
#		preprocessing_train_file = ('s_%d_train.csv' % (subject))
#		preprocessed = pd.DataFrame(index=idx, columns=columns, data=x_train)
#		preprocessed.to_csv(preprocessing_train_file, index_label='id', float_format='%.6f')

		
		#mse = MSE( y, pred_tot )
		#rmse = sqrt( mse )
		#print "testing RMSE:", rmse
	
	x_train = np.array(x_train).astype(float)
	y_train = np.array(y_train).astype(int)
	
	
	zeroes = np.array(y_train)
	for i in range(0, 119496):
		for j in range (0, 6):
			zeroes[i, j] = 0
			
	mse = MSE( zeroes, y_train )
	rmse = m.sqrt( mse )
	print ("zeroes RMSE:", rmse)
	

	#We can now quickly sample a training set while holding out 40% of the data for testing (evaluating) our classifier:
	X_train, X_test, y_tr, y_te = cross_validation.train_test_split(x_train, y_train, test_size=0.4, random_state=0)

from sklearn import svm
svc = svm.SVC(kernel='rbf')
clf = svc.fit(X_train, y_tr[:, 0])
print ( 'RBF train score: ' + str(clf.score(X_train, y_tr[:, 0])) )
print ( 'test score: ' + str(clf.score(X_test, y_te[:, 0])) )

svc2 = svm.SVC(kernel='poly', degree=3)
clf3 = svc2.fit(X_train, y_tr[:, 0])
print ( 'POLY train score: ' + str(clf3.score(X_train, y_tr[:, 0])) )
print ( 'test score: ' + str(clf3.score(X_test, y_te[:, 0])) )

from sklearn import neighbors
knn = neighbors.KNeighborsClassifier()
clf2 = knn.fit(X_train, y_tr[:, 0])
print ( 'KNN train score: ' + str(clf2.score(X_train, y_tr[:, 0])) )
print ( 'test score: ' + str(clf2.score(X_test, y_te[:, 0])) )


