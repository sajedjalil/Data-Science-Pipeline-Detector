# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
from time import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, lda,
                     random_projection)
t0 = time()                     
# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")

# preprocessing columns
# factorizing column values
train['T1_V4'] = pd.factorize(train['T1_V4'])[0]
train['T1_V5'] = pd.factorize(train['T1_V5'])[0]
# simple yes/no
train['T1_V6'] = pd.factorize(train['T1_V6'])[0]
train['T1_V7'] = pd.factorize(train['T1_V7'])[0]
train['T1_V8'] = pd.factorize(train['T1_V8'])[0]
train['T1_V9'] = pd.factorize(train['T1_V9'])[0]
train['T1_V11'] = pd.factorize(train['T1_V11'])[0]
train['T1_V12'] = pd.factorize(train['T1_V12'])[0]
train['T1_V15'] = pd.factorize(train['T1_V15'])[0]
train['T1_V16'] = pd.factorize(train['T1_V16'])[0]
train['T1_V17'] = pd.factorize(train['T1_V17'])[0]

train['T2_V3'] = pd.factorize(train['T2_V3'])[0]
train['T2_V5'] = pd.factorize(train['T2_V5'])[0]
train['T2_V11'] = pd.factorize(train['T2_V11'])[0]
train['T2_V12'] = pd.factorize(train['T2_V12'])[0]
train['T2_V13'] = pd.factorize(train['T2_V13'])[0]

#train.to_csv('data/train_numeric.csv')
color = train["Hazard"]
del train["Hazard"]
del train["Id"]
X = train.as_matrix()

t1 = time()
print("Computing T-SNE projection")

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
Y = tsne.fit_transform(X[:1000,:])
t2 = time()
print("t-SNE: %.2g sec" % (t2 - t1))
print("length of Y: " + str(len(Y)))

for i in range(1,50):
    t1 = time()
    Y = np.vstack((Y,tsne.fit_transform(X[i*1000:(i+1)*1000,:])))
    t2 = time()
    print(str(i) + "th iter t-SNE: %.2g sec" % (t2 - t1))
    print("length of Y: " + str(len(Y)))
    
t1 = time()
Y = np.vstack((Y,tsne.fit_transform(X[50000:,:])))
t2 = time()
print("t-SNE: %.2g sec" % (t2 - t1))
print("length of Y: " + str(len(Y)))

fig = plt.figure(figsize=(15, 8))
plt.scatter(Y[:, 0], Y[:, 1],c=color[:],cmap=plt.cm.spectral)

t3 = time()
print("overall: %.5g sec" % (t3 - t0))

plt.tight_layout()

plt.savefig("t-SNE_train.png")