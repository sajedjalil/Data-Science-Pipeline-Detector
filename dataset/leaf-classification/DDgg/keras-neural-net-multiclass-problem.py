## -------------------------------------------------------------------------------------------
## Keras-Neural-Net-multiclass-problem // runtime 90 seconnds // LB score: 
##
## https://www.kaggle.com/tobikaggle/leaf-classification/keras-neural-net-multiclass-problem/
## modified fork from forks of forks
## https://www.kaggle.com/najeebkhan/leaf-classification/neural-network-through-keras/notebook
## https://www.kaggle.com/apapiu/leaf-classification/neural-network-through-keras/
## https://www.kaggle.com/zenstat/leaf-classification/nn-through-keras-copied/
## https://www.kaggle.com/tobikaggle/leaf-classification/nn-through-keras-copied-mod
## -------------------------------------------------------------------------------------------

## import standard libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## import TensorFlow and SciKit learn
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

## import Keras Neural Network
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.utils.np_utils import to_categorical

# print OS and python info
import sys
print(sys.version)
print('pandas', pd.__version__)

# read data
data = pd.read_csv('../input/train.csv')
parent_data = data.copy()  
ID = data.pop('id')

## since the labels are textual, so we encode them categorically
y = data.pop('species')
y = LabelEncoder().fit(y).transform(y)

## most of the learning algorithms are prone to feature scaling
## standardising the data to give zero mean =)
X = StandardScaler().fit(data).transform(data)
print(X.shape)

## we will be working with categorical crossentropy function
## it is required to further convert the labels into "one-hot" representation
y_cat = to_categorical(y)
print(y_cat.shape)

## developing a KERAS  layered model for Neural Networks
## input dimensions should be equal to the number of features
## we used softmax layer (99 classes) to predict a uniform probabilistic distribution of outcomes
## bigher network topology will increase training time and can lead to overfitting
## best to perform grid search to find optimal parameters
model = Sequential()
model.add(Dense(2048,input_dim=192,  init='uniform', activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1024, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(99, activation='softmax'))
print('Computing...')

## Error is measured as categorical crossentropy or multiclass logloss
## for multiclass models 'categorical_crossentropy' is recommended
## optimizers are listed here https://keras.io/optimizers/ 
## optimizer=Adagrad, rmsprop, SGD, Adadelta, Adam, Adamax, Nadam 
## AdaMax -->  Validation accuracy : 0.686868686869 ; 2:03 min (2 epochs)
## Adam -->    Validation accuracy : 0.656565656566 ; 2:41 min (2 epochs)
## Adagrad --> Validation accuracy : 0.767676767677 ; 1:44 min (2 epochs)
np.random.seed(12345)
model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ["accuracy"])

## fitting the model on the whole training data
## increasing epoch size can improve accuracy (see plot)
## increase epoch size to get higher accuracy, but also longer run time too
## for this problem nb_epoch=60 will be efficient enough
## introduce early stopping (not implemented here)
## see graph of epochs vs accuracy or crossentropy here
## https://www.kaggle.com/tobikaggle/leaf-classification/nn-through-keras-copied-mod/run/472153/
history = model.fit(X,y_cat,batch_size=1, nb_epoch=20,verbose=0, validation_split=0.1)

## print validation accuracy, higher is better; no guarantee to avoid overfitting
## we need to consider the loss for final submission to leaderboard
print(history.history.keys())
print('val_acc: ',min(history.history['val_acc']))
print('val_loss: ',min(history.history['val_loss']))
print('acc: ',min(history.history['acc']))
print('loss: ',min(history.history['loss']))

## read test sed and fit model
test = pd.read_csv('../input/test.csv')
index = test.pop('id')
test = StandardScaler().fit(test).transform(test)
yPred = model.predict_proba(test)

## converting the test predictions in a dataframe as depicted by sample submission
yPred = pd.DataFrame(yPred,index=index,columns=sorted(parent_data.species.unique()))
fp = open('submission_keras_NN.csv','w')
fp.write(yPred.to_csv())

## END 



