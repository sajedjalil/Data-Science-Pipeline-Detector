# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

"""KERNEL TO COMPARE THE ACCURACY OF DIFFERENT MODELS ON THE GHOSTS,GOBLINS AND GHOULS! CLASSIFICATION COMPETITION """

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


#Path to the datasets
TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"

#Reading in the train and test data into dataframes
trainset = pd.read_csv(TRAIN_PATH)
testset = pd.read_csv(TEST_PATH) 

#Exploring the data
print(trainset.shape)
print(trainset.isnull().any())
print(testset.isnull().any())


#Separating the dependent variable from the independent variables
y_train = trainset.iloc[:,-1]
print(y_train)

x_train = trainset.iloc[:,1:-1]
print(x_train)

#For encoding the categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
labelencoder_COL= LabelEncoder()
y_train= labelencoder_Y.fit_transform(y_train)  
#print(y_train)
x_train['color']=labelencoder_COL.fit_transform(x_train['color'])
#print(x_train)

#Checking correlation between the independent variables
print(trainset.corr()) #not much correlation amongst the variables


#For getting the accuracy of the trained models 
from sklearn import metrics

#TRAINING THE DATA ON DIFFERENT MODELS

#Naive-Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_train)
print(metrics.accuracy_score(y_train,y_pred))
#0.749 accuracy

#Multi-Layer Perceptron
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000 )
clf.fit(x_train, y_train) 
y_predmlp = clf.predict(x_train)
print(metrics.accuracy_score(y_train,y_predmlp))
#0.722 accuracy

#Gradient Boosting
from xgboost.sklearn import XGBClassifier
clfX=XGBClassifier()
clfX.fit(x_train, y_train) 
y_predX = clfX.predict(x_train)
print(metrics.accuracy_score(y_train,y_predX))
#0.938 accuracy


#For multi-class encoding
from keras.utils import to_categorical
encoded = to_categorical(y_train)
print(encoded)

#Artificial Neural Network 
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing the ANN 
clfann= Sequential()
clfann.add(Dense(output_dim=10,init='uniform',activation='relu',input_dim=5)) #1st hidden layer
clfann.add(Dense(output_dim=10,init='uniform',activation='relu'))#2st hidden layer
clfann.add(Dense(output_dim=3,init='uniform',activation='sigmoid'))#3st hidden layer
clfann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
clfann.fit(x_train,encoded,batch_size=10,nb_epoch=200) 
y_predann= clfann.predict(x_train) 
#0.793 accuracy

#I havnt really played around with the model parameters to get a better classification, feel free to do so youself.