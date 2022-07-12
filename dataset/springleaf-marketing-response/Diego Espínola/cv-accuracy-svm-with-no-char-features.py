
# -*- coding: UTF-8 -*-
#enconding: utf-8
"""
@author: Diego Espínola
"""

#-------------------------------------------------------------------------------
""" CLASS """
class Timer:
    
    def __init__(self):
        self.t_start=0
        self.t_end=0
        
    def startTimer(self):
        import time
        self.t_start = time.time()
        print ('Timer starts')
        
    def endTimer(self,display):
        import time
        import math
        tend = time.time() - self.t_start
        mi=math.floor(tend/60)
        sec=round(tend%60)        
        if display==True:
            s=('EXE TIME: %02d min' %mi)+(' %2.1f sec' %sec)
            print (s)
        self.t_end=tend
        return mi,sec
    

""" METHODS """
def folder(X,y,N_fold,Interleaving,disp):

    if Interleaving==1:
        if disp==1:
            print ('Interleaving')

        n_sample = len(X)
        import numpy as np
        np.random.seed(0)
        order = np.random.permutation(n_sample)
        X = X[order]
        y = y[order]

    if disp==1:
        print ('Creating the %d-Folds' %N_fold)
    fold=len(X)/N_fold
##    print fold
    X_fold=[]
    y_fold=[]
    for i in range(N_fold-1):
##        print i
##        print X[i*fold:(i+1)*fold]
        X_fold.append(X[i*fold:(i+1)*fold])
        y_fold.append(y[i*fold:(i+1)*fold])
    
##    print N_fold
##    print X[(N_fold-1)*fold:]
    X_fold.append(X[(N_fold-1)*fold:])
    y_fold.append(y[(N_fold-1)*fold:])
    
    return X_fold,y_fold

def trainTest(X_fold,y_fold,i,N_fold,disOut):
    if disOut==1:
        print ('Creating the X_test and X_train -> N_fold=%d (of %d):' %(i+1, N_fold))
    X_test  = []
    y_test  = []
    X_train = []
    y_train = []

    for ii in range(N_fold):
        if ii==i:
            for iii in range(len(X_fold[ii])):
                X_test.append(X_fold[ii][iii])
                y_test.append(y_fold[ii][iii])
        else:
            for iii in range(len(X_fold[ii])):
                X_train.append(X_fold[ii][iii])
                y_train.append(y_fold[ii][iii])
    
    return X_train,y_train,X_test,y_test

#-------------------------------------------------------------------------------
""" MAIN """

import pandas as pd
import numpy as np
import math
import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
mytimer=Timer()
mytimer.startTimer()

""" START MAIN """
print('START_MAIN_EXE')

# READ DATA -------
print ('** READ DATA **')

data = pd.read_csv("../input/train.csv",index_col=0)
print ('n_rows = %d' %(len(data)))

y=data['target']
y=np.array(y)
del data['target']

ones=0
zeros=0
for i in range(len(y)):
	if y[i]==1:
		ones=ones+1
	if y[i]==0:
		zeros=zeros+1
print ('N class TRAIN')
print ('class 1 = %d' %ones)
print ('class 0 = %d' %zeros)
print ('N_features = ',len(data.columns))

#PRE DATA -------
print ('** F.E DATA **')

#----remove fields no char and fileds with same NaN
dt=data.dtypes
colName=data.columns
anyNAN=data.isnull().any()

for i in range(len(colName)):
	checkFormat=((dt[i]=='int64')or(dt[i]=='int32')or(dt[i]=='float64')or(dt[i]=='float32'))
	if ((anyNAN[i]==True)or(checkFormat==False)):
		del data[colName[i]]
	del checkFormat
del dt,anyNAN

print ('N_features = ',len(data.columns))

X=np.array(data)
del colName,data


#----we take the same nummber of trial in class 0 and class 1
if ones<zeros:
    nclass=ones
if ones>zeros:
    nclass=zeros
nclass1=0
nclass0=0
XX=[]
yy=[]
for i in range(len(X)):
    if nclass1<nclass:
        if y[i]==1:            
            nclass1=nclass1+1
            XX.append(X[i])
            yy.append(y[i])
    if nclass0<nclass:
        if y[i]==0:
            nclass0=nclass0+1
            XX.append(X[i])
            yy.append(y[i])

X=XX
y=yy
del XX,yy
X=np.array(X)
y=np.array(y)

#----As the data train is so much porcessing info, we take same random trial
print ('n_rows = %d' %(len(X)))
N=len(X)
por=0.1
i=1
XX=[]
yy=[]
while (i<N):
    a=np.random.rand()
    if a<por:
        XX.append(X[i])
        yy.append(y[i])
    i=i+1
del N,i,por,a


print ('n_rows = %d (Randomed trial)' %(len(XX)))
X=XX
y=yy
del XX,yy
X=np.array(X)
y=np.array(y)


ones=0
zeros=0
for i in range(len(y)):
	if y[i]==1:
		ones=ones+1
	if y[i]==0:
		zeros=zeros+1
print ('N class TRAIN')
print ('class 1 = %d' %ones)
print ('class 0 = %d' %zeros)
print ('N_features = ',len(X[0]))

#----Remove features with low variance

from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X=sel.fit_transform(X)
print ('N_features = ',len(X[0]))
del sel

#----Normalization
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X)
X=scaler.transform(X)
del scaler

print ('Summary dimension = ',len(X),'*',len(X[0]))

# CROSS VALIDATION -------
print ('** CROSS VALIDATION **')

n_sample = len(X)
import numpy as np
np.random.seed(0)
order = np.random.permutation(n_sample)
X = X[order]
y = y[order]

N_fold=2
disp=1
Interleaving=1
X_fold,y_fold=folder(X,y,N_fold,Interleaving,disp)

# CROSS VALIDATION SVM -------
print ('** CV SVM **')

from sklearn.svm import SVC
kernelSVM=['linear','rbf']
##C_SVM=[0.1,10]
C_SVM=[1]
accFinal=[]
for iii in range(len(kernelSVM)):       
	for ii in range(len(C_SVM)):
		print ('Kernel=%s and C=%1.1f):' %(kernelSVM[iii],C_SVM[ii]))
		acc=[]
		for i in range(N_fold):
			X_train,y_train,X_test,y_test=trainTest(X_fold,y_fold,i,N_fold,disp)
			clf = SVC(C=C_SVM[ii],kernel=kernelSVM[iii],probability=False)
			clf.fit(X_train,y_train)
			y_pred=clf.predict(X_test)
			boleanData=y_pred==y_test
			acc.append(boleanData.mean())
			del clf,y_pred,boleanData
		acc=np.array(acc)
		acc=acc.mean()
		print ('Accuracy = %1.3f:' %(acc))
		accFinal.append(acc)
    
""" END MAIN """
print('END_MAIN_EXE')
mytimer.endTimer(display=True)
del mytimer
