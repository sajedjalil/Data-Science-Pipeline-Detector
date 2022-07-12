# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:01:21 2016

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


import pandas as pd

import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation
from sklearn import ensemble


print('Load data...')
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

target = train['target'].values
labels = train["target"]
id_test = test['ID'].values
trainId = target
testId = test["ID"]

train = train.drop(['ID','target','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)
test = test.drop(['ID','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)

#test = test.drop(['ID'],axis=1)
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




train = np.asarray(train, dtype=np.float32)        
labels = np.asarray(labels, dtype=np.int32).reshape(-1,1)

net = NeuralNet(
    layers=[  
        ('input', InputLayer),
        ('dropout0', DropoutLayer),
        ('hidden1', DenseLayer),
        ('dropout1', DropoutLayer),
        ('hidden2', DenseLayer),
        ('output', DenseLayer),
        ],

    input_shape=(None, len(train[1])),
    dropout0_p=0.1,
    hidden1_num_units=50,
    hidden1_W=Uniform(),
    dropout1_p=0.2, 
    hidden2_num_units=40,
    #hidden2_W=Uniform(),

    output_nonlinearity=sigmoid,
    output_num_units=1, 
    update=nesterov_momentum,
    update_learning_rate=theano.shared(np.float32(0.01)),
    update_momentum=theano.shared(np.float32(0.9)),    
    # Decay the learning rate
    on_epoch_finished=[AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
                       AdjustVariable('update_momentum', start=0.9, stop=0.99),
                       ],
    regression=True,
    y_tensor_type = T.imatrix,                   
    objective_loss_function = binary_crossentropy,
    #batch_iterator_train = BatchIterator(batch_size = 256),
    max_epochs=20, 
    eval_size=0.1,
    #train_split =0.0,
    verbose=2,
    )
    
print(net.batch_iterator_train)

seednumber=1235
np.random.seed(seednumber)
net.fit(train, labels)


preds = net.predict_proba(test)[:,0] 


submission = pd.read_csv('../input/sample_submission.csv')
submission["PredictedProb"] = preds
submission.to_csv('NNbench.csv', index=False)

