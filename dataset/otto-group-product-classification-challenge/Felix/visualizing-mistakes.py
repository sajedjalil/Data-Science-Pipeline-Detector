import pandas as pd
import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import StratifiedKFold

import matplotlib.pyplot as plt
import seaborn

import xgboost as xgb

from math import log

# The idea of this script is to train a quick model on the Otto data and then ask in particular, 
# can we tell what's going on when the model misclassifies things?
#
# If we can tell that a data point has been misclassified, then maybe that information can be useful
# in adjusting the probability estimates.
#
# This is a last-minute idea that never made it into our final submissions, so it seems like a fun thing to
# try in a post-competition script.
# (the xgboost/modelling part of this script is based on the XGBoost benchmarks posted by TomHall and followed up by Chris)

#seaborn.set_style("whitegrid")
#seaborn.set_context("notebook", font_scale = 1.5)

dataframe = pd.read_csv("../input/train.csv")

LE = LabelEncoder()
DV = DictVectorizer()

# The log isn't important for xgboost, but it will help with the PCA later on
data = np.log(2.0 + DV.fit_transform(dataframe.iloc[:,1:94].T.to_dict().values()).todense())
LE.fit(["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6", "Class_7", "Class_8", "Class_9"])
labels = LE.transform(dataframe.iloc[:,94])
idx = np.array(dataframe.iloc[:,0])

# The data is all sorted, which causes problems for XGBoost, so lets shuffle.
shuf = np.random.permutation(labels.shape[0])
idx = idx[shuf]
labels = labels[shuf]
data = data[shuf,:]

# Not the best parameters, but hopefully fast enough to run on Kaggle

param = {'max_depth':10, 'min_loss_reduction': 0.6, 'min_child_weight':6, 'subsample': 0.7, 'eta':0.3, 'silent':1, 'objective':'multi:softprob', 'nthread':4, 'num_class': 9 }

SKF = StratifiedKFold(n_folds = 3, y = labels)
train_preds = np.zeros( (0,9) )

for tridx,tsidx in SKF:
    dtrain = xgb.DMatrix(data[tridx], label = labels[tridx])
    dtest = xgb.DMatrix(data[tsidx])
    bst = xgb.train( param, dtrain, 20 )
    train_preds = np.append( train_preds, bst.predict(dtest), axis=0 )

err = 0
for i in range(train_preds.shape[0]):
    err -= log(1e-15 + train_preds[i, labels[i]])

err /= train_preds.shape[0]

print("CV log-loss of the base estimator: {}".format(err))

# What we really wanted to get was the misclassified rows, however.
# Lets see what makes us incorrectly classify things into a particular class
for examine_class in range(9): # Look at each class 
    threshold = 0.3 # Pick out rows that have P>threshold for this class, incorrectly
    
    # These are all the data rows that aren't members of the class we're examining
    baseidx = (labels != examine_class)
    
    # Pick out the rows that have erroneously high predictions
    wrongidx = train_preds[baseidx, examine_class] >= threshold
    # ...and everything else
    rightidx = train_preds[baseidx, examine_class] < threshold
    
    wrongdata = (data[baseidx,:])[wrongidx,:]
    okaydata = (data[baseidx,:])[rightidx,:]
    
    # I want a 2D visualization here, so I'm going to do LDA to separate the good predictions 
    # from the bad predictions in 1D, then add one of the PCA components for clarity
    lda = LDA()
    pca = PCA()
    
    wrongproj = pca.fit_transform(wrongdata) # Fit a projection specifically on the error rows
    okayproj = pca.transform(okaydata) # Also project down the data which is not incorrectly classified in this fashion at least
    
    # I need labels for LDA
    wl = np.zeros( wrongproj.shape[0] )
    wl[:] = 1
    ol = np.zeros( okayproj.shape[0] )
    ol[:] = 0
    pcl = np.append(wl, ol)
    
    lda.fit(np.append(wrongdata, okaydata, axis=0), pcl)
    wrong_ld = lda.transform(wrongdata)
    okay_ld = lda.transform(okaydata)
    
    plt.plot(okay_ld[:,0],okayproj[:,0], 'o', markersize=4, alpha=0.2, label="Not Mistaken", color="#3030a0")
    plt.plot(wrong_ld[:,0],wrongproj[:,0], 'o', markersize=6, label="Mistaken", alpha=0.8, color="#f03030", markeredgecolor='#600000', markeredgewidth=1)
    plt.xlabel("LDA Feature")
    plt.ylabel("PCA Feature")
    plt.ylim( min( okayproj[:,0].min(), wrongproj[:,0].min() )-1, max( okayproj[:,0].max(), wrongproj[:,0].max() )+2 )
    plt.title("Misclassification as Class "+str(examine_class+1))
    plt.legend()
    plt.savefig("class"+str(examine_class+1)+".png")
    plt.clf()