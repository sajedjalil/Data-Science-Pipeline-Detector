import pandas as pd
import os
import numpy as np
import scipy
import datetime
import csv
import random
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
from lasagne.updates import nesterov_momentum
from lasagne.objectives import binary_crossentropy
from nolearn.lasagne import NeuralNet
import theano
from theano import tensor as T
from theano.tensor.nnet import sigmoid
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn import ensemble, svm, neighbors, linear_model, cross_validation, preprocessing
import seaborn as sns
import keras

date = datetime.datetime.now()

if date.day == 17:

    with open("bestsubmission.csv","w") as outfile:
        for e, line in enumerate(open("../input/sampleSubmission.csv")):
            if e == 0:
                outfile.write(line)
            else:
                r = line.strip().split(",")
                outfile.write("%s,%s\n"%(r[0],random.random()))
else:
    print("Come back in {} days.".format(17 - date.day))
