#Without argpartition Script with all features used took 4 days but with argpartition it executes in 3 hours on Core 2 Duo

#Best Value of K

# Prime Nearest Number(Number of Features)^1/2 With K as a random guess...

#This is Sample Script
import pandas as pd
import os
import numpy as np

os.system("ls ../input")

data = pd.read_csv("../input/train.csv")
data.describe()
data.columns


X=np.asarray(data.ix[:,1:-1].dropna(),dtype=np.float32)
print (X.shape)
Y=np.asarray(data.ix[:,-1])
print (Y, Y.shape, len(np.unique(Y))) # so we have 9 classes...

#print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

#print(train.head())
#datatest=pd.read_csv('../input/test.csv')
#ids=np.array(datatest['id'])
#Xtest=np.array(datatest.ix[:,1:].values,dtype=np.float32)

#feat=np.arange(X.shape[1])
#knn1=KNearestNeighbor(11)# Prime 93^1/2 With K as a random guess...
#knn1.train(X[:,feat],Y)
#pclassesNewWith11=knn1.predict(Xtest[:,feat])

# value of K if used 11 or 7 the best possible accuracy with KNN can be achieved. 