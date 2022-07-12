# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import manifold # manifold learning methods
from sklearn import svm # support vector machine
import matplotlib.pyplot as plt # plotting
# keras for neural network stuff:
from keras.models import Model
from keras.layers import Dense,Dropout,Input
from keras import regularizers
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#---------------------------
# read and preprocess data
#---------------------------
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
samplesub = pd.read_csv('../input/sample_submission.csv')

x = np.float32(train.values[:,2:])
x = np.concatenate((x,np.float32(test.values[:,1:])),axis=0)
# scale to 0-1 range
x-=np.nanmin(x,axis=0)
x/=np.nanmax(x,axis=0)
labels = train.values[:,1]
# get labels into number form
unique_labels = samplesub.columns.values[1:]#list(set(labels))
print('There are ',len(unique_labels),'unique labels/species and ',len(x),'data points')
label_dict = dict(zip(unique_labels,range(len(unique_labels))))  # name to number
nlabel_dict = dict(zip(range(len(unique_labels)),unique_labels)) # number to name
nlabels = np.array([label_dict[l] for l in labels])
colors = ['r','g','b','m','k','c']

#------------------------------------
# Fitting Isomap and plotting results
#------------------------------------
print('fitting isomap...')
ismp = manifold.Isomap(n_neighbors=10,n_components=2)
ismp.fit(x)

plt.figure(1)
ax = plt.subplot(111)
y1 = ismp.transform(x)
plt.scatter(y1[990:,0],y1[990:,1],marker='.')
z=0
for p1, p2 in zip(y1[:990,0], y1[:990,1]):
    plt.text(p1, p2, str(nlabels[z]),fontsize=8,color=colors[nlabels[z]%6])#, color="red")#, fontsize=12)
    z+=1
#for i, txt in enumerate(nlabels[::2]):
#    ax.annotate(txt, (y1[i,0],y1[i,1]),fontsize=8)
plt.title('Isomap embedding')
plt.savefig('plot2.png')

#-----------------------------------
# Fitting t-SNE and plotting results
#-----------------------------------
plt.figure(2)
ax = plt.subplot(111)
print('fitting t-SNE...')
tsne = manifold.TSNE(n_components=2,init='pca')
y2 = tsne.fit_transform(x)
plt.scatter(y2[990:,0],y2[990:,1],marker='.')
z=0
for p1, p2 in zip(y2[:990,0], y2[:990,1]):
    plt.text(p1, p2, str(nlabels[z]),fontsize=8,color=colors[nlabels[z]%6])#, color="red")#, fontsize=12)
    z+=1
#for i, txt in enumerate(nlabels[::2]):
#    ax.annotate(txt, (y2[i,0],y2[i,1]),fontsize=8)
plt.title('t-SNE embedding')
plt.savefig('plot1.png')


#---------------------------------------
# Can we use the manifold embeddings for
# classification?
#---------------------------------------
print('fitting SVMs...')
svm_clf1 = svm.SVC(probability=True)
svm_clf1.fit(y1[:990,:],nlabels[:990])
svm_clf2 = svm.SVC(probability=True)
svm_clf2.fit(y2[:990,:],nlabels[:990])
#-----------------------------------------
# visualize separation
#-----------------------------------------
# "scan" whole input range
xx1,yy1 = np.meshgrid(np.arange(np.nanmin(y1[:,0]),np.nanmax(y1[:,0]),.5),
                    np.arange(np.nanmin(y1[:,1]),np.nanmax(y1[:,1]),.5))
whole_space1 = np.concatenate((xx1.flatten()[:,None],yy1.flatten()[:,None]),axis=1)
xx2,yy2 = np.meshgrid(np.arange(np.nanmin(y2[:,0]),np.nanmax(y2[:,0]),.5),
                    np.arange(np.nanmin(y2[:,1]),np.nanmax(y2[:,1]),.5))
whole_space2 = np.concatenate((xx2.flatten()[:,None],yy2.flatten()[:,None]),axis=1)

# get predictions for whole input space: Isomap
plt.figure(4)
color = np.float32( svm_clf1.predict(whole_space1) )
plt.pcolormesh(xx1,yy1,color.reshape(xx1.shape),cmap=plt.cm.bone)
plt.scatter(y1[990:,0],y1[990:,1],marker='.',label='unlabeled')
z=0
for p1, p2 in zip(y1[:990,0], y1[:990,1]):
    plt.text(p1, p2, str(nlabels[z]),fontsize=8,color=colors[nlabels[z]%6])#, color="red")#, fontsize=12)
    z+=1
plt.legend(prop={'size':6})
plt.xlim([np.nanmin(y1[:,0]),np.nanmax(y1[:,0])])
plt.ylim([np.nanmin(y1[:,1]),np.nanmax(y1[:,1])])
plt.title('RBF-SVM classification: ISOMAP embedding')
plt.savefig('separation_isomap.png')
# get predictions for whole input space: t-SNE
plt.figure(5)
color = np.float32( svm_clf2.predict(whole_space2) )
plt.pcolormesh(xx2,yy2,color.reshape(xx2.shape),cmap=plt.cm.bone)
plt.scatter(y2[990:,0],y2[990:,1],marker='.',label='unlabeled')
z=0
for p1, p2 in zip(y2[:990,0], y2[:990,1]):
    plt.text(p1, p2, str(nlabels[z]),fontsize=8,color=colors[nlabels[z]%6])#, color="red")#, fontsize=12)
    z+=1
plt.legend(prop={'size':6})
plt.xlim([np.nanmin(y2[:,0]),np.nanmax(y2[:,0])])
plt.ylim([np.nanmin(y2[:,1]),np.nanmax(y2[:,1])])
plt.title('RBF-SVM classification: t-SNE embedding')
plt.savefig('separation_tSNE.png')
#-----------------------------------------
# How does a neural network change the 
# t-SNE embedding?
#-----------------------------------------
epochs = 15
# 0: center data and prepare output
x-=.5

def onehot(X):
    X_1hot = np.zeros((len(X),np.nanmax(X)+1))
    for k in range(len(X)):
        X_1hot[k,X[k]] = 1
    return X_1hot
    
y1h = onehot(nlabels)   # one hot representation for output
# 1: set up a neural network
inp = Input(shape=(x.shape[1],))
D1 = Dropout(.01)(inp)
L1 = Dense(1024, init='uniform', activation='tanh',activity_regularizer=regularizers.activity_l1(.01))(D1)
L2 = Dense(len(unique_labels), init='uniform', activation='softmax')(L1)
# 1.1: the model we train to classify the data
model1 = Model(inp,L2)
# 1.2: model for getting output from layer 1
repNN1 = Model(inp,L1)
# 1.3: compile all models
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy','binary_crossentropy'])
repNN1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy','binary_crossentropy'])
# check if there are existing weight of if we need to train
weight_name = 'NN_sparse_rep_H1_1024.h5'
try:
    model1.load_weights(weight_name)
    print('Yay! found existing weights...')
except IOError:
    model1.fit(x[:990,:], y1h[:990,:], nb_epoch=epochs, batch_size=15)
    model1.save_weights(weight_name,overwrite=True)
# 2: let's take a look at the output
NN_rep1 = repNN1.predict(x)
NN_rep1 = tsne.fit_transform(NN_rep1)

#---------------------------------------------------
# We did it for the t-SNE embedding of the raw data
# so let's also do it for the neural network 
# t-SNE embedding of the layer 1 activations:
# classify using an RBF SVM
#---------------------------------------------------
plt.figure(6)
svm_clf3 = svm.SVC(probability=True)
svm_clf3.fit(NN_rep1[:990,:],nlabels[:990])
print('SVM scores t-SNE: ',svm_clf2.score(y2[:990,:],nlabels))
print('SVM scores NN & t-SNE: ',svm_clf3.score(NN_rep1[:990,:],nlabels))
xx3,yy3 = np.meshgrid(np.arange(np.nanmin(NN_rep1[:,0]),np.nanmax(NN_rep1[:,0]),.5),
                    np.arange(np.nanmin(NN_rep1[:,1]),np.nanmax(NN_rep1[:,1]),.5))
whole_space3 = np.concatenate((xx3.flatten()[:,None],yy3.flatten()[:,None]),axis=1)
color = np.float32( svm_clf3.predict(whole_space3) )
plt.pcolormesh(xx3,yy3,color.reshape(xx3.shape),cmap=plt.cm.bone)
plt.scatter(NN_rep1[990:,0],NN_rep1[990:,1],marker='.')
z=0
for p1, p2 in zip(NN_rep1[:990,0], NN_rep1[:990,1]):
    plt.text(p1, p2, str(nlabels[z]),fontsize=8,color=colors[nlabels[z]%6])#, color="red")#, fontsize=12)
    z+=1
plt.xlim([np.nanmin(NN_rep1[:,0]),np.nanmax(NN_rep1[:,0])])
plt.ylim([np.nanmin(NN_rep1[:,1]),np.nanmax(NN_rep1[:,1])])
plt.title('t-SNE embedding of NN hidden layer activations (1024 units, sparse)')
plt.savefig('tSNE_neural_network.png')