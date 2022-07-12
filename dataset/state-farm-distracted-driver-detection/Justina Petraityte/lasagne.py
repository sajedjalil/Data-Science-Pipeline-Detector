# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import scipy
import glob #path patter expression
import os
import cv2
import matplotlib.pyplot as plt

from cv2 import cvtColor
import numpy as np

import matplotlib.pyplot as plt
from sklearn import svm
#import lasagne # deep learning library
import PIL
from PIL import Image
import theano.tensor as T
import lasagne
import theano


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


imgs_list = pd.read_csv('../input/driver_imgs_list.csv')
class_number = imgs_list['classname'].str.extract('(\d)', expand =False) #convert class name to number


#C:/Users/Juste/Desktop/Kaggle/imgs/train/*/*.jpg

train_files = [f for f in glob.glob("../input/train/*/*.jpg")]
#train_files = [f for f in glob.glob("C:/Users/Juste/Desktop/Kaggle/imgs/train/*/*.jpg")]
labels = pd.DataFrame({'label': ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'], 'class_name':['safe driving','texting - right', 'talking on the phone - right', 'texting - left', 'talking on the phone - left', 'operating the radio', 'drinking', 'reaching behind', 'hair and makeup', 'talking to passenger' ]})

data = pd.DataFrame({'file':train_files})

data['label'] = data['file'].str.extract('(c.)', expand=True)

train = data.merge(labels, on = 'label', how = 'left')



read = []
train1 = train.iloc[0:300,:]

for i in range(0,len(train1)):
    read.append(list((Image.open(train1.iloc[i,0]).getdata())))
    

pix1 = pd.DataFrame({'image':read[:]})
train2 = pd.concat([train1, pix1], axis = 1)

print(train2)

input_var = T.tensor4('X')
target_var = T.ivector('y')



from lasagne.nonlinearities import leaky_rectify, softmax
network = lasagne.layers.InputLayer((None, 3, 32, 32), input_var)
network = lasagne.layers.Conv2DLayer(network, 64, (3, 3),
                                     nonlinearity=leaky_rectify)
network = lasagne.layers.Conv2DLayer(network, 32, (3, 3),
                                     nonlinearity=leaky_rectify)
network = lasagne.layers.Pool2DLayer(network, (3, 3), stride=2, mode='max')
network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                    128, nonlinearity=leaky_rectify,
                                    W=lasagne.init.Orthogonal())
network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                    10, nonlinearity=softmax)



# create loss function
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean() + 1e-4 * lasagne.regularization.regularize_network_params(
        network, lasagne.regularization.l2)

# create parameter update expressions
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01,
                                            momentum=0.9)

# compile training function that updates parameters and returns training loss
train_fn = theano.function([input_var, target_var], loss, updates=updates)



x_train = train2.iloc[:,3]
y_train = train2.iloc[:,1]
