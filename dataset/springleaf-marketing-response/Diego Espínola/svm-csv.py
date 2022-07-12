# -*- coding: UTF-8 -*-
#enconding: utf-8
"""
@author: Diego Espínola
"""
##""" TIMER """
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
            print(s)
        self.t_end=tend
        return mi,sec
        
""" MAIN ------------------------------------------------------------------- """

import pandas as pd
import numpy as np
import math
import os
import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
mytimer=Timer()
mytimer.startTimer()


""" START MAIN """
print('START_GEN_CSV_EXE')

print('** READ DATA **')

data=pd.read_csv("../input/train.csv",index_col=0)
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

colName=data.columns

X=np.array(data)
del data


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


#----Normalization
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X)
X=scaler.transform(X)


print ('Summary dimension train_data= ',len(X),'*',len(X[0]))


print ('** TRAINING SVM **')
from sklearn.svm import SVC
kernelSVM='rbf'
C_SVM=1
clf = SVC(C=C_SVM,kernel=kernelSVM,probability=False)
#clf.fit(X,y)
del X,y

# READ DATA TEST
X_test=pd.read_csv("../input/test.csv",index_col=0)
X_test=X_test[colName]
ids=X_test.index.tolist()
X_test=np.array(X_test)
X_test=sel.transform(X_test)
#X_test=scaler.transform(X_test)
print ('Summary dimension train_data= ',len(X_test),'*',len(X_test[0]))
del sel
#del scaler


#pred=clf.predict(X_test)
#del X_test,X

#submission_file='SVMlinear_hard.csv'
##CSV
# create pandas object for sbmission
#submission = pd.DataFrame(index=ids,columns=['target'],data=pred)
#submission.to_csv(submission_file,index_label='id',float_format='%1.3f')
#del submission


""" END MAIN """
print('END_GEN_CSV_EXE')
mytimer.endTimer(display=True)
del mytimer
