#! /usr/bin/env python

# Calculates the median from only a subset of data points
# Some basic ways to maybe improve:
# 1) Try out different treatments of NA values
# 2) Add in some day-to-day variation in predictions
#
# Beyond that, probably not much can be done to improve median-based 
# methods significantly more. I would guess that egression and things like ARIMA
# models would be the next step to get significantly better.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def validation(data,npts,test=60):
# Get the SMAPE metric for this entry given a number of different models
# Test on (default) 60 days at the end of the data
# 
  smape = np.zeros(len(npts))
  prediction = np.zeros(len(npts))
  first_nonzero = len(data)-test
  for i in range(len(data)-test):
    if data[i] > 0:
      first_nonzero = i
      break
  for i,n in enumerate(npts):
    start = np.max([first_nonzero,len(data) - n-test])
    if start < len(data)-test:
      prediction[i] = np.median(data[start:len(data)-test])

  for i in range(test):
    truth = data[i+len(data)-test]
    for j in range(len(npts)):
      if truth!=0 or prediction[j]!=0:
        smape[j] += np.abs(truth - prediction[j]) / \
                    (np.abs(truth)+np.abs(prediction[j]))

  return smape

def make_prediction(data):
# Actually predict the future data
# The validations shows looking at the median of
# the last 25 pts should work best
  prediction = 0
  first_nonzero = len(data)
  for i in range(len(data)):
    if data[i] > 0:
      first_nonzero = i
      break

  start = np.max([first_nonzero,len(data)-25])
  if start < len(data):
    prediction = np.median(data[start:])
  months={'2017-01':31,'2017-02':28,'2017-03':1}
  data = {}
  for key in months:
    for day in range(1,months[key]+1):
      newkey = key+'-'+"%02i"%(day)
      data[newkey] = prediction

  return data

def run_analysis():

# First read the training set
  print('Reading training set')
  train = pd.read_csv('../input/train_1.csv').fillna(0)
  print('Downcasting to int')
  for col in train.columns[1:]:
    train[col] = pd.to_numeric(train[col],downcast='integer')
  print('Training: ')
  print(train.info())
  keys = pd.read_csv('../input/key_1.csv',index_col=0)
  print('Key: ')
  print(keys.info())



  pages = train['Page']
  # Get just the data
  train = train.iloc[:,1:]
  with open('output.csv','w') as outfile:

    outfile.write('Id,Visits\n')
    for i in range(train.shape[0]):
      if i%1000==0:
        print('On event %i of %i'%(i,train.shape[0]))
      data = train.iloc[i,:]
      pred = make_prediction(data)
      for key in pred:
        newkey = "%s_%s" %(pages.iloc[i],key)
        outkey = keys.loc[newkey,'Id']
        outfile.write(outkey+','+str(pred[key])+'\n')

def run_validation():
# Check a bunch of different models and print the results from 
# excluding the last 60 days from the training set.
# Should really implement this with something like a 10-fold CV
# in order to get some sense of the uncertainty

# First read the training set
  print('Reading training set')
  train = pd.read_csv('../input/train_1.csv').fillna(0)
  print('Downcasting to int')
  for col in train.columns[1:]:
    train[col] = pd.to_numeric(train[col],downcast='integer')
  print('Training set info: ')
  print(train.info())
  print('done')


  pages = train['Page']
  pts = [1,15,25,35,50,75,100,125,150,200,250]
  smape = np.zeros(len(pts))
  train = train.iloc[:,1:]
  for i in range(train.shape[0]):
# for loop here is ugly and also pretty slow
    if i%1000==0:
      print('On event %i of %i'%(i,train.shape[0]))
    data = train.iloc[i,:]
    smape += validation(data,pts)

  smape *= 200. / (60*train.shape[0])
  for i in range(len(pts)):
    print('Start pos: %i SMAPE: %f'%(pts[i],smape[i]))


  plt.plot(pts,smape)
  plt.ylabel('SMAPE')
  plt.xlabel('# of Samples from end of data used')
  plt.title('Validation curve for a median-based analysis')
  plt.show()

if __name__=="__main__":
  run_analysis()
