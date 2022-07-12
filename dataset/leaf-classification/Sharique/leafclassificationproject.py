# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import csv as csv 
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats  

#The sklearn libaries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier

#The keras libraries
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.utils.np_utils import to_categorical


traindf = pd.read_csv('../input/train.csv', header=0) 
x_train = traindf.drop(['id', 'species'], axis=1)
scaler = StandardScaler().fit(x_train) #to standardize values before feeding into the model
x_train = scaler.transform(x_train) 

# encoding categorical fields
label_encoder = LabelEncoder().fit(traindf['species'])
y_train = label_encoder.transform(traindf['species'])


test_data = pd.read_csv('../input/test.csv')
test_id = test_data.pop('id')
x_test = test_data.values
x_test = scaler.transform(x_test)

#-----------LogisticRegression--------------------

params = {'C':[100, 1000], 'tol': [0.001, 0.0001]}
#We initialise the Logistic Regression
lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
# We use grid search to find the best values for C and tol and initialize the model with them
gs = GridSearchCV(lr, params, scoring=None, refit='True', cv=3) 
gs_model = gs.fit(x_train, y_train)

y_test = gs.predict_proba(x_test)

submissionLR = pd.DataFrame(y_test, index = test_id, columns = label_encoder.classes_)
submissionLR.to_csv('submissionsLR.csv')

#----------------------------------

#-----------SVM--------------------

svc = SVC(kernel="rbf", C=0.025, probability=True)
svc_model = svc.fit(x_train, y_train)

y_test = svc.predict_proba(x_test)

submissionSVM = pd.DataFrame(y_test, index = test_id, columns = label_encoder.classes_)
submissionSVM.to_csv('submissionsSVM.csv')

#----------------------------------

#-----------Random Forest--------------------

rf = ExtraTreesClassifier(n_estimators=500, random_state=0)
rf_model = rf.fit(x_train, y_train)

y_test = rf.predict_proba(x_test)

submissionRF = pd.DataFrame(y_test, index = test_id, columns = label_encoder.classes_)
submissionRF.to_csv('submissionsRF.csv')

#----------------------------------

#-----------DecisionTree--------------------

dt = DecisionTreeClassifier()
dt_model = dt.fit(x_train, y_train)

y_test = dt.predict_proba(x_test)

submissionDT = pd.DataFrame(y_test, index = test_id, columns = label_encoder.classes_)
submissionDT.to_csv('submissionsDT.csv')

#----------------------------------

#------------NaiveBayes-------------------

nb = GaussianNB()
nb_model = nb.fit(x_train, y_train)

y_test = nb.predict_proba(x_test)

submissionNB = pd.DataFrame(y_test, index = test_id, columns = label_encoder.classes_)
submissionNB.to_csv('submissionsNB.csv')

#----------------------------------

#------------NeuralNetwork-------------------

model = Sequential() 
model.add(Dense(2000,input_dim=192, init='uniform', activation='relu')) 
model.add(Dropout(0.3)) 
model.add(Dense(1000, activation='sigmoid')) 
model.add(Dropout(0.3)) 
model.add(Dense(99, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop')
y_train = to_categorical(y_train)
history = model.fit(x_train,y_train,batch_size=128,nb_epoch=500,verbose=0)

y_test = model.predict_proba(x_test)

submissionNN = pd.DataFrame(y_test, index = test_id, columns = label_encoder.classes_)
submissionNN.to_csv('submissionsNN.csv')

#----------------------------------