# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# -*- coding: utf-8 -*-
"""
@author: Octavio González-Lugo
"""

###############################################################################
#                          Packages to use 
###############################################################################

import numpy as np
import pandas as pd

from sklearn.svm import NuSVC
from scipy.spatial import distance as ds
from sklearn.covariance import LedoitWolf

###############################################################################
#                          Loading the data 
###############################################################################

TrainData=pd.read_csv('../input/train.csv')
DataColumns=[c for c in TrainData.columns if c not in ['id','target','wheezy-copper-turtle-magic']]

Xtrain=TrainData
Xtest=pd.read_csv('../input/test.csv')

###############################################################################
#                          General Functions 
###############################################################################

#Calculate the covariance matrix and the mean of each cluster  
def GetModelParams(DataFrame,ColumnIndex):
  
  cDataSet=DataFrame
  
  cData0=cDataSet[cDataSet['target']==0]
  cData1=cDataSet[cDataSet['target']==1]
  
  bData0=np.array(cData0[ColumnIndex])
  bData1=np.array(cData1[ColumnIndex])
  
  Cov0=LedoitWolf(assume_centered=False).fit(bData0)
  Cov1=LedoitWolf(assume_centered=False).fit(bData1)
  
  Mean0=bData0.mean(axis=0)
  Mean1=bData1.mean(axis=0)
  
  return Cov0.covariance_,Cov1.covariance_,Mean0,Mean1

#Calculate the coeficients to calculate the kullback-leiber divergence
def KullbackLeiberCoefficients(CovarianceA,CovarianceB):
  
  invA=np.linalg.inv(CovarianceA)
  coef1a=np.dot(CovarianceB,invA)
  _,coefa=np.linalg.slogdet(coef1a)
  coef1b=np.dot(invA,CovarianceB)
  coefb=np.trace(coef1b)
  
  return invA,coefa,coefb

#KL divergence and Total variation distance 
def KullbackLeiberDivergence(CoefficientA,CoefficientB,CoefficientC,Mean,Sample):
  
  distance=(ds.mahalanobis(Mean,Sample,CoefficientA))**2
  divergence=CoefficientC+distance-CoefficientB-len(Mean)
  
  return np.sqrt(divergence/2)

#Calculation of the constants from 
def MakeModelParams(Data,ColumnIndex):
  
  cData=Data
  Cov0,Cov1,Mean0,Mean1=GetModelParams(cData,ColumnIndex)
  Inv0,CoefA0,CoefB0=KullbackLeiberCoefficients(Cov0,Cov1)
  Inv1,CoefA1,CoefB1=KullbackLeiberCoefficients(Cov1,Cov0)
  
  return Mean0,Mean1,Inv0,CoefA0,CoefB0,Inv1,CoefA1,CoefB1

#Sample features 
def SampleDivergence(Sample,Params):
  
  cSample=Sample
  Mean0,Mean1=Params[0],Params[1]
  Inv0,CoefA0,CoefB0=Params[2],Params[3],Params[4]
  Inv1,CoefA1,CoefB1=Params[5],Params[6],Params[7]
  
  div00=KullbackLeiberDivergence(Inv0,CoefA0,CoefB0,np.array(Mean0),cSample)
  div01=KullbackLeiberDivergence(Inv0,CoefA0,CoefB0,np.array(Mean1),cSample)
  div10=KullbackLeiberDivergence(Inv1,CoefA1,CoefB1,np.array(Mean0),cSample)
  div11=KullbackLeiberDivergence(Inv1,CoefA1,CoefB1,np.array(Mean1),cSample)
  
  return [div00/div10,div01/div11,div00/div01,div10/div11,(div00-div10)/div10,(div00-div01)/div01,(div01-div11)/div11,(div10-div11)/div11]

#Calculate the features for all the samples 
def ModelFeatures(Data,Params,ColumnIndex):
  
  cData=Data
  trainData=np.array(cData[ColumnIndex])
  container=[]
  
  for k in range(len(trainData)):
    
    cSample=trainData[k]
    container.append(SampleDivergence(cSample,Params))
    
  return np.array(container)
    
###############################################################################
#                          Training the models  
###############################################################################

nModels=512

ParamLib=[]
LinModels=[]
ModelColumns=[]

for k in range(nModels):
  
  train2 = Xtrain[Xtrain['wheezy-copper-turtle-magic']==k]
  Vars=train2.std(axis=0)
  VarColumns=[Vars.index[k] for k in range(len(Vars)) if Vars.iloc[k]>1.5]
  ModelColumns.append(VarColumns)
  
  Params=MakeModelParams(train2,VarColumns)
  ParamLib.append(Params)
  
  mfeat=ModelFeatures(train2,Params,VarColumns)
  labls=np.array(train2['target'])
  
  clr = NuSVC(gamma='scale',kernel='poly',degree=3,probability=True,random_state=256)
  clr.fit(mfeat,labls)

  LinModels.append(clr)
 
###############################################################################
#                          Submit predictions  
###############################################################################

Predictions=[]

for j in range(len(Xtest)):
  
  ModelNumber=Xtest.iloc[j]['wheezy-copper-turtle-magic']
  ModelData=np.array(Xtest.iloc[j][ModelColumns[ModelNumber]],dtype=np.float64)
  
  cDivergence=np.array(SampleDivergence(ModelData,ParamLib[ModelNumber]))
  cPred=LinModels[ModelNumber].predict_proba(cDivergence.reshape(1,-1))
  Predictions.append([Xtest.iloc[j]['id'],cPred[0][1]])

Predictions=np.array(Predictions)
submission=pd.read_csv('../input/sample_submission.csv')

submission['id']=Predictions[:,0]
submission['target']=Predictions[:,1]
submission.to_csv('submission.csv',index=False)
