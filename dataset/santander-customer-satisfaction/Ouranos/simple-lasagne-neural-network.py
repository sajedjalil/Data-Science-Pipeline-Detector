# -*- coding: utf-8 -*-
"""
Created on Tue Mar 7 

@author: Ouranos
"""


import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
from lasagne.updates import nesterov_momentum
from lasagne.objectives import binary_crossentropy
from lasagne.init import Uniform
from nolearn.lasagne import NeuralNet, BatchIterator
import theano
from theano import tensor as T
from theano.tensor.nnet import sigmoid



class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()       
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


        
def pdFillNAN(df, strategy = "mean"):
    #Fills empty values with either the mean value of each feature, or an indicated number
    if strategy == "mean":
        return df.fillna(df.mean())
    elif type(strategy) == int:
        return df.fillna(strategy)



train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Impute -1 to NAs
train.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)


np.random.seed(3210)
train = train.iloc[np.random.permutation(len(train))]

#Drop columns with 0 variation
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)

print(len(remove))
train.drop(labels = remove, axis=1, inplace=True)
test.drop(labels = remove, axis=1, inplace=True)


# remove duplicated columns
remove = []
c = train.columns
for i in range(len(c)-1):
    v = train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,train[c[j]].values):
            remove.append(c[j])
train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

#Drop target, ID and some other columns...
labels = train["TARGET"]
trainId = train["ID"]
testId = test["ID"]

train.drop(labels = ["ID","TARGET"], axis = 1, inplace = True)
test.drop(labels =["ID"], axis = 1, inplace = True)


print ("Scaling...")
train, scaler = preprocess_data(train)
test, scaler = preprocess_data(test, scaler)


train = np.asarray(train, dtype=np.float32)        
labels = np.asarray(labels, dtype=np.int32).reshape(-1,1)

net = NeuralNet(
    layers=[  
        ('input', InputLayer),
        ('hidden1', DenseLayer),
        ('hidden2', DenseLayer),
        ('hidden3', DenseLayer),
        ('output', DenseLayer),
        ],

    input_shape=(None, len(train[1])),
    hidden1_num_units=100,
    hidden1_W=Uniform(), 
    hidden2_num_units=50,
    hidden2_W=Uniform(),
    hidden3_num_units=25,
    hidden3_W=Uniform(),
    
    output_nonlinearity=sigmoid,
    output_num_units=1, 
    update=nesterov_momentum,
    update_learning_rate=theano.shared(np.float32(0.001)),
    update_momentum=theano.shared(np.float32(0.9)),    
    # Decay the learning rate
    on_epoch_finished=[AdjustVariable('update_learning_rate', start=0.001, stop=0.0001),
                       AdjustVariable('update_momentum', start=0.9, stop=0.99),
                       ],
    regression=True,
    y_tensor_type = T.imatrix,                   
    objective_loss_function = binary_crossentropy,
    #batch_iterator_train = BatchIterator(batch_size = 256),
    max_epochs=30, 
    eval_size=0.0,
    verbose=2,
    )


seednumber=1235
np.random.seed(seednumber)
net.fit(train, labels)


preds = net.predict_proba(test)[:,0] 


submission = pd.read_csv('../input/sample_submission.csv')
submission["TARGET"] = preds
submission.to_csv('Lasagne_bench.csv', index=False)
