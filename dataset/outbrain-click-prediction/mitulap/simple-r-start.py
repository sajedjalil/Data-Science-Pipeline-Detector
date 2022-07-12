import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import gc # We're gonna be clearing memory a lot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, linear_model
from sklearn.naive_bayes import GaussianNB
le = preprocessing.LabelEncoder()

X=[]
XT=[]
Y=[]
bufsize = 65536
i = 0
with open('../input/clicks_train.csv') as infile: 
    while True:
        lines = infile.readlines(bufsize)
        if not lines:
            break
        for line in lines:
            if i ==0 :
                i=i+1
                continue
            if i > 10000:
                break
            x_i=[]
            linee = line.split(',')
            x_i.append(linee[0])
            x_i.append(linee[1])
            x_i=np.array(x_i).astype(np.int)
            x_i.reshape(1, 2)
            Y.append(linee[2])
            X.append(x_i)
            i=i+1
i = 0           
with open('../input/clicks_test.csv') as infile: 
    while True:
        lines = infile.readlines(bufsize)
        if not lines:
            break
        if i > 1000:
           break
        for line in lines:
            if i==0 :
                i=i+1
                continue
            x_i=[]
            linee = line.split(',')
            if len(linee) < 2:
                continue
            x_i.append(linee[0])
            x_i.append(linee[1])
            XT.append(x_i)
            #print ("Hey",len(XT))
            i=i+1            
#infile.close()
#print X,Y
#data = df.ix[:,:].values
print (len(X))
print (len(XT))
X= np.array(X).astype(np.int)
XT=np.array(XT).astype(np.int)
Y= np.array(Y).astype(np.int)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
X=X.reshape(len(X), 2)
Y=Y.reshape(len(Y), 1)
XT=XT.reshape(len(XT), 2)
# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(X,Y)
gnb.fit(X,Y.ravel())
slope = regr.coef_[0][0]
intercept = regr.intercept_
# The coefficients
print("y = %f + %f " %( intercept,slope))
# The mean squared error
print("Mean squared error: %.10f"
     % np.mean((regr.predict(X) -Y) ** 2))
i=0    
for x in XT:
    if i > 100:
        break
    xp=np.array(x).astype(np.int)
    xp=xp.reshape(1, 2)
    print (xp)
    print(regr.predict(xp))
    i=i+1