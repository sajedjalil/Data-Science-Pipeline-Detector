import pandas as pd
import zipfile
import numpy as np
import matplotlib.pyplot as pl
import csv
import sys

z=zipfile.ZipFile('../input/train.csv.zip')

train=pd.read_csv(z.open('train.csv'),parse_dates=['Dates'])
train['Month']=train['Dates'].dt.month
train['Year']=train['Dates'].dt.year
train['Hour']=train['Dates'].dt.hour
train['Street']=pd.Series(train['Address']).str.replace("/"," ")
#print(train.head(20))

uniqueResolution=pd.Series.unique(train['Resolution'])
uniquePD=pd.Series.unique(train['PdDistrict'])
#print(len(uniqueDescription)) , 879
#print(len(train)), 878049
print(uniquePD[:5])
#print(uniqueResolution)

#pl.scatter(train[''],train[])
"Creating test and training dataset"

#msk=np.random.rand(len(train))<0.7
#train_model=train[msk]
#test_model=train[~msk]

