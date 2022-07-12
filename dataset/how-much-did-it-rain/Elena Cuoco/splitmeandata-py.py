"""
How much did it Rain?
"""

# Authors: Elena Cuoco <elena.cuoco@gmail.com>
#
#Kaggle How much did it rain competition using pandas and scikit library
import numpy as np
import pandas as pd
from datetime import datetime, date, time
 
from pandas.io.common import ZipFile 
 
#json library for settings file
import json,csv
import sys,os,shutil
##Read configuration parameters

#file_dir = 'Settings-wk.json'
#config = json.loads(open(file_dir).read())

#test_file=config["HOME"]+config["TEST_DATA_PATH"]+'test_2014.csv'
#train_file=config["HOME"]+config["TRAIN_DATA_PATH"]+'train_2013.csv'
#trainfile= config["HOME"]+config["TRAIN_DATA_PATH"]+'train_1.csv'
#testfile= config["HOME"]+config["TEST_DATA_PATH"]+'test_1.csv'
#test_file='../input/test_2014.csv.zip' 
train_file='../input/train_2013.csv.zip'
trainfile= './train_1.csv'
#testfile=  './test_1.csv'
###############################################################################
# Main
###############################################################################
#1126695  141729447 1060745948 train_2013.csv
#630453  85951587 637487640 test_2014.csv

header_train=['Id','TimeToEnd', 'DistanceToRadar', 'Composite', 'HybridScan', 'HydrometeorType', 'Kdp', 'RR1', 'RR2', 'RR3', 'RadarQualityIndex', \
'Reflectivity', 'ReflectivityQC', 'RhoHV', 'Velocity', 'Zdr', 'LogWaterVolume', 'MassWeightedMean', 'MassWeightedSD', 'Expected']
header0=['TimeToEnd', 'DistanceToRadar', 'Composite', 'HybridScan', 'HydrometeorType', 'Kdp', 'RR1', 'RR2', 'RR3', 'RadarQualityIndex', \
'Reflectivity', 'ReflectivityQC', 'RhoHV', 'Velocity', 'Zdr', 'LogWaterVolume', 'MassWeightedMean', 'MassWeightedSD']
header_test=['Id','TimeToEnd', 'DistanceToRadar', 'Composite', 'HybridScan', 'HydrometeorType', 'Kdp', 'RR1', 'RR2', 'RR3', 'RadarQualityIndex', \
'Reflectivity', 'ReflectivityQC', 'RhoHV', 'Velocity', 'Zdr', 'LogWaterVolume', 'MassWeightedMean', 'MassWeightedSD']

def train_data(data):
  df = pd.DataFrame(columns=header_train)
  kk=0
  for i in range(len(data)):
     a=np.asarray(str(data.iloc[i,1]).split(' ')).astype(float)
     len2add=len(a)
     for k in range(len2add):
      df.loc[kk+k,'Id']=data.Id[i]
      df.loc[kk+k,'Expected']=data.Expected.values[i]
     for el in header0:
      a=np.asarray(str(data.loc[i,el]).split(' ')).astype(float)
      for k in range(len(a)):
       df.loc[kk+k,el] = a[k]
     kk+=len2add
  df.fillna(0.0)
  df.replace(to_replace='NaN', value=0.0, inplace=True)
  df.replace(to_replace='nan', value=0.0, inplace=True)
  df.replace(to_replace='999.0', value=0.0, inplace=True)
  df.replace(to_replace='-99900.0', value=0.0, inplace=True)
  df.replace(to_replace='-99901.0', value=0.0, inplace=True)
  df.replace(to_replace='-99903.0', value=0.0, inplace=True)
  df.replace(to_replace='-999.0', value=0.0, inplace=True)
  df.replace(to_replace='99000.0', value=0.0, inplace=True)
  df.replace(to_replace='99901.0', value=0.0, inplace=True)
  df.replace(to_replace='99903.0', value=0.0, inplace=True)

  df2=df.groupby('Id').aggregate(np.mean)
  return df2

def test_data(data):
  df = pd.DataFrame(columns=header_test)
  kk=0
  for i in range(len(data)):
     a=np.asarray(str(data.iloc[i,1]).split(' ')).astype(float)
     len2add=len(a)
     for k in range(len2add):
      df.loc[kk+k,'Id']=data.Id[i]
     for el in header0:
      a=np.asarray(str(data.loc[i,el]).split(' ')).astype(float)
      for k in range(len(a)):
       df.loc[kk+k,el] = a[k]
  #   kk+=len2add
  df.fillna(0.0)
  df.replace(to_replace='NaN', value=0.0, inplace=True)
  df.replace(to_replace='nan', value=0.0, inplace=True)
  df.replace(to_replace='999.0', value=0.0, inplace=True)
  df.replace(to_replace='-99900.0', value=0.0, inplace=True)
  df.replace(to_replace='-99901.0', value=0.0, inplace=True)
  df.replace(to_replace='-99903.0', value=0.0, inplace=True)
  df.replace(to_replace='-999.0', value=0.0, inplace=True)
  df.replace(to_replace='99000.0', value=0.0, inplace=True)
  df.replace(to_replace='99901.0', value=0.0, inplace=True)
  df.replace(to_replace='99903.0', value=0.0, inplace=True)

  df2=df.groupby('Id').aggregate(np.mean)
  return df2
###################################################################################
# start training
with ZipFile(train_file) as z:
     f = z.open('train_2013.csv')
  

 
#the commented lines avoid the script to run on all data and so exceed the running time 
#reader = pd.read_table(train_file, sep=',', chunksize=1000, names=header_train,header=0)
reader = pd.read_table(f, sep=',', names=header_train,header=0,iterator=True )
with open(trainfile, 'a') as outfile:
#    i=0
#    for data in reader:
      data=reader.get_chunk(100)
      df=train_data(data)

     # if i==0:
      df.to_csv(outfile)
      #else:
      #  df.to_csv(outfile,header=None)
     # i+=1
     # if i%10==0:
      #  print(i)

print ('Ended train data preparation')

 