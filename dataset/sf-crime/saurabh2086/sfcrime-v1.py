#Importing Libraries
import sklearn
import numpy as np
import scipy as sp
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import time
import zipfile

#Importing training Data
crimeData = pd.read_csv('../input/train.csv')

# Converiting text into datetime
#crimeData['Datetime'] = crimeData['Dates'].apply(lambda x: time.strptime(x,'%Y-%m-%d %H:%M:%S'))
crimeData['Dates'] = pd.to_datetime(crimeData['Dates'])

crimeData['year'] = crimeData['Dates'].apply(lambda x: x.year)
crimeData['Month'] = crimeData['Dates'].apply(lambda x: x.month)
crimeData['Hour'] = crimeData['Dates'].apply(lambda x: x.hour)

def bincreation(x):
    y = ""
    if 0< x <= 4:
        y= "0-4"
    elif 4< x <= 8:
        y = "4-8"
    elif 8< x <= 12:
        y = "8-12"
    elif 12 <= x <= 16:
        y = "12-16"
    elif 16< x <= 20:
        y = "16-20"
    elif 20 < x <= 24:
        y = "20-24"
    return y
    
crimeData['HourBin'] = crimeData['Hour'].apply(lambda x: bincreation(x))

#print(crimeData.groupby('Hour').size())

print(set(crimeData.HourBin))

#print(crimeData.head())
#Creating dummy variables.
crimeData = pd.concat([crimeData, pd.get_dummies(crimeData.PdDistrict), pd.get_dummies(crimeData.DayOfWeek), pd.get_dummies(crimeData.year),pd.get_dummies(crimeData.Month),pd.get_dummies(crimeData.HourBin)], axis = 1)
crime = crimeData.drop(['Dates','Descript','DayOfWeek','PdDistrict','Resolution','Address','year','X','Y','Month','Hour','HourBin'],axis=1)
#print(crime.head())

msk = sp.random.rand(len(crime))<0.8
trainCrime = crime[msk]
testCrime = crime[~msk]

train_y = trainCrime['Category'].values
train_x = np.array(trainCrime.drop('Category',axis=1))

from sklearn.neighbors import KNeighborsClassifier
crimeModel = KNeighborsClassifier(n_neighbors=1)
crimeModel.fit(train_x,train_y)

test_y = testCrime['Category'].values
test_x = np.array(testCrime.drop('Category',axis=1))
print(crimeModel.score(test_x,test_y))
#print(crimeData.head())
# Exploringtype of crime and rest of the Data
#print(crimeData.groupby('Category').size())
#print(crimeData.describe(include = 'all'))
#crimeCat = list(set(crimeData.Category))
##plotting the Data on map


#Extracting latitude range
#lat_min = min(crimeData.X)
#lat_max = max(crimeData.X)

#print(np.mean(crimeData.X))

#Extracting logitude 
#lon_min = min(crimeData.Y)
#lon_max = max(crimeData.Y)

#importing MAp Data
#mapdata = np.loadtxt("../input/sf_map_copyright_openstreetmap_contributors.txt")
 


