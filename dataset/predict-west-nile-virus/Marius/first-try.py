import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
weather=pd.read_csv("../input/weather.csv")
sample = pd.read_csv('../input/sampleSubmission.csv')

weather_stn1 = weather[weather['Station']==1]
weather_stn2 = weather[weather['Station']==2]
weather_stn1 = weather_stn1.drop('Station', axis=1)
weather_stn2 = weather_stn2.drop('Station', axis=1)
weather = weather_stn1.merge(weather_stn2, on='Date')

tag=['Tmax','Tmin','Tavg','Depart','DewPoint','WetBulb','Heat','Cool','Sunrise','Sunset','CodeSum','Depth','Water1','SnowFall','PrecipTotal','StnPressure','SeaLevel','ResultSpeed','ResultDir','AvgSpeed']
#to get useful tag with low miss
#for i in tag:
#    print(i)
#    print(weather[i].value_counts())
#print(weather['Id'].value_counts())
UsefulTag=['Date','Tmax_y','Tmin_y','Tavg_y','DewPoint_y','WetBulb_y','Heat_y','Cool_y','PrecipTotal_y','StnPressure_y','ResultSpeed_y','ResultDir_y','AvgSpeed_y','Tmax_x','Tmin_x','Tavg_x','DewPoint_x','WetBulb_x','Heat_x','Cool_x','PrecipTotal_x','StnPressure_x','ResultSpeed_x','ResultDir_x','AvgSpeed_x']

UsefulWeather=weather[UsefulTag]
#print(UsefulWeather)
#print(weather.columns)

#print(UsefulTag[2:])

#for u in UsefulTag[2:]:
#    print(u)
#    print(UsefulWeather[UsefulWeather[u]>=0].describe())

UsefulWeather = UsefulWeather.replace('M', 0)
UsefulWeather = UsefulWeather.replace('-', 0)
UsefulWeather = UsefulWeather.replace('T', 0)
UsefulWeather = UsefulWeather.replace(' T', 0)
UsefulWeather = UsefulWeather.replace('  T', 0)

train = train.merge(UsefulWeather, on='Date')
test = test.merge(UsefulWeather, on='Date')

#get index of columns
#index1=train.columns
#index2=test.columns
#print(index1,index2)
#for i in index1:
#    print(i)
#    print(train[i].value_counts())
#for j in index2:
#    print(j)
#    print(train[j].value_counts())

#a=train['NumMosquitos'].groupby(train['WnvPresent'])
#print(a.mean())
#x0=train[train['WnvPresent']==0]
#y0=x0['NumMosquitos']
#x1=train[train['WnvPresent']==1]
#y1=x1['NumMosquitos']
#print(y0.describe())
#print(y1.describe())
#print(weather)
#print(train)
#print(test)

def create_month(x):
    return x.split('-')[1]

train['month'] = train.Date.apply(create_month)
test['month'] = test.Date.apply(create_month)

#drop=['Date','Address','Species','Block','Street','Trap','AddressNumberAndStreet','Latitude','Longitude','AddressAccuracy']
train=train.drop(['Date','Address','Species','Block','Street','Trap','AddressNumberAndStreet','Latitude','Longitude','AddressAccuracy'],axis=1)
test=test.drop(['Id','Date','Address','Species','Block','Street','Trap','AddressNumberAndStreet','Latitude','Longitude','AddressAccuracy'],axis=1)
#train['try']=1

#print(train)
PreKey=['NumMosquitos','WnvPresent']
Pre=train[PreKey]
train=train.drop(['NumMosquitos','WnvPresent'],axis=1)
#print(Pre)
use=['Tavg_y','WetBulb_y','Heat_y','Cool_y','StnPressure_y','AvgSpeed_y','Tavg_x','WetBulb_x','Heat_x','Cool_x','StnPressure_x','AvgSpeed_x','month']
train=train[use]
test=test[use]

#index1=train.columns
#index2=test.columns
#print(index1,index2)

#from sklearn.tree import DecisionTreeClassifier
#rgb = DecisionTreeClassifier(max_depth=None, min_samples_split=1,random_state=0)
#rgb.fit(train,Pre['WnvPresent'])
#print(rgb.score(train,Pre['WnvPresent']))
#out=rgb.predict(test)
#out[:,0].to_csv('first.csv',index=False)
#print(np.shape(out))
#out.shape=(116293,1)
#out=np.transpose(out)
#sample['WnvPresent']=out[:,1]
#print(sample.shape())
#print(sample)
#sample.to_csv('a.csv',index=False)
#print(np.shape(out))
#np.savetxt('a.txt',out)
#from sklearn import linear_model
#clf = linear_model.LogisticRegression()
#print(Pre[Pre['WnvPresent']==1].describe())
#print(Pre[Pre['WnvPresent']==0].describe())
#print(clf.score(Pre['NumMosquitos'],Pre['WnvPresent']))
#from sklearn import linear_model
#clf = linear_model.BayesianRidge()
#clf.fit(train,Pre['NumMosquitos'])
#print(clf.score(train,Pre['NumMosquitos']))
#z=clf.predict(test)
#print(z)
#q=[]
#for i in range(len(z)):
#    if z[i]>17:
#        q.append(1)
#    else:
#        q.append(0)
#print(q,len(q),len(z))
#f=open('d.txt','a')
#for j in range(len(q)):
#    f.write(str(j))
#    f.write(',')
#    f.write(str(q[j]))
#    f.write(',')
#    f.write('\n')
#f.close()
#import csv
#csvfile=file('a.csv','wb')
"""
from sklearn import ensemble
clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=10, min_samples_split=1)
clf.fit(train, Pre['WnvPresent'])
print(clf.score(train, Pre['WnvPresent']))
predictions = clf.predict_proba(test)[:,1]
sample['WnvPresent'] = predictions
sample.to_csv('submit.csv', index=False)
"""
from sklearn.linear_model import SGDClassifier
clf =  SGDClassifier(loss="hinge", penalty="l2")
clf.fit(train, Pre['WnvPresent'])
print(clf.score(train, Pre['WnvPresent']))
predictions = clf.predict(test)
sample['WnvPresent'] = predictions
sample.to_csv('submit.csv', index=False)
