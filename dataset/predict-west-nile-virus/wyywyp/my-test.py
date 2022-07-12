from __future__ import print_function
import numpy as np
import datetime
import csv
import os
from sklearn import svm ,grid_search
import sklearn.metrics as metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import pandas as pd
import time 
from sklearn.ensemble import RandomForestClassifier

'''
def precip(text):
    TRACE = 1e-3
    text = text.strip()
    if text == "M":
        return None
    if text == "T":
        return TRACE
    return float(text)
'''
def closest_station(lat, long):
    # Chicago is small enough that we can treat coordinates as rectangular.
    stations = np.array([[41.995, -87.933],
                         [41.786, -87.752]])
    loc = np.array([lat, long])
    deltas = stations - loc[None, :]
    dist2 = (deltas**2).sum(1)
    return np.argmin(dist2)
       

def replaceText(T):
	TRACE = 1e-3
	for i in range(len(T.index)):
		for j in range(len(T.columns)):		
			if T.iat[i,j]=='M':
				T.iat[i,j]=None
			if T.iat[i,j]=='  T':
				T.iat[i,j]=TRACE

def judge(x,word):
	y=0
	if word in x:
		y=1
	return y

def judge_empty(x):
	y=0
	if len(x)==1:
		y=1
	return y

def inverse(line):
	# inverse True and False in a list
	for i,x in enumerate(line):
		if x == True:
			line[i]=False
		if x == False:
			line[i]=True
	return line

def impute_missing_weather_station_values(weather1,weather2):
    # Stupid simple
    j=weather1.index.isin(weather2.index)
    j=inverse(j)
    weather2.append(weather1[j])

    j=weather2.index.isin(weather1.index)
    j=inverse(j)
    weather1.append(weather2[j])
    
    for i in weather1.index:
    	for j in weather1.columns:
    		if weather1.loc[i,j] == None:
    			weather1.loc[i,j]=weather2.loc[i,j]
    		if weather2.loc[i,j] == None:
    			weather2.loc[i,j]=weather1.loc[i,j]

def scaled_count(record):
	SCALE = 10.0
	return int(np.ceil(record["NumMosquitos"] / SCALE))

def assemble_X(base, weather1,weather2):
	X=[]
	for i in base.index:
		b=base.ix[i,:]
		date = b["Date"]
		lat,lng=b['Latitude'],b['Longitude']
		case=[b['Date'].year,b['Date'].month,b['Date'].day,lat,lng]
		case.extend(b[['CULEX_P','CULEX_R']].values)

		station = closest_station(lat, lng)
		
		for days_ago in [1,3,7,14]:
			day = date - datetime.timedelta(days=days_ago)
			for obs in ["Tmax" ,"Tmin" ,"Tavg" ,"DewPoint" , "WetBulb" ,"PrecipTotal" ,"Depart" , "ResultSpeed" ,"ResultDir" ,"AvgSpeed",'RA','BR','TSRA','TS','HZ','DZ','FG','VCTS','CODE_E']:
				if station == 0:
					case.append(weather1.ix[day,obs])
				else:
					case.append(weather2.ix[day,obs])
		for repeat in range(scaled_count(b)):
			X.append(case)
	X=np.asarray(X, dtype=np.float32)
	return X
					
def assemble_y(base):
	y = []
	for i in base.index:
		b=base.ix[i,:]
		present = base.ix[i,"WnvPresent"]
		for repeat in range(scaled_count(b)):
			y.append(present)    
	return np.asarray(y, dtype=np.int32).reshape(-1,1)


def report(test_lab,p_lab):
    if sum(p_lab)==0 or sum(p_lab)==len(p_lab):
        print (' fail with sum of p_lab :',sum(p_lab))

    print ('classification_report :')
    print (metrics.classification_report(test_lab,p_lab))
            
    print ('the f1_score :',metrics.f1_score(test_lab,p_lab))
    
    print ('AUC_score :',metrics.roc_auc_score(test_lab,p_lab))


print ("beging ",round(time.clock()))

cwd=os.getcwd()
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
weather=pd.read_csv('../input/weather.csv')
sample = pd.read_csv('../input/sampleSubmission.csv')

#pocess the training set
train=train[['Date','Latitude','Longitude','Species',"NumMosquitos","WnvPresent"]]
train['Date']=pd.to_datetime(train['Date'])
# SPECIES OF MOSQUE
train['CULEX_P']=train['Species'].apply(judge,word='CULEX PIPIENS')
train['CULEX_R']=train['Species'].apply(judge,word='CULEX RESTUANS')


##pocess the test set

test=test[['Date','Latitude','Longitude','Species']]
test["NumMosquitos"]=1
test['Date']=pd.to_datetime(test['Date'])
# SPECIES OF MOSQUE
test['CULEX_P']=test['Species'].apply(judge,word='CULEX PIPIENS')
test['CULEX_R']=test['Species'].apply(judge,word='CULEX RESTUANS')

#pocess the weather
title=["Station","Date" ,"Tmax" ,"Tmin" ,"Tavg" ,
"DewPoint" , "WetBulb" ,"PrecipTotal" ,"Depart" , 
"ResultSpeed" ,"ResultDir" ,"AvgSpeed" ,"StnPressure", "SeaLevel" ,
'CodeSum']

weather=weather[title]
weather['RA']=weather['CodeSum'].apply(judge,word='RA')
weather['BR']=weather['CodeSum'].apply(judge,word='BR')
weather['TSRA']=weather['CodeSum'].apply(judge,word='TSRA')
weather['TS']=weather['CodeSum'].apply(judge,word='TS')
weather['HZ']=weather['CodeSum'].apply(judge,word='HZ')
weather['DZ']=weather['CodeSum'].apply(judge,word='DZ')
weather['FG']=weather['CodeSum'].apply(judge,word='FG')
weather['VCTS']=weather['CodeSum'].apply(judge,word='VCTS')
weather['CODE_E']=weather['CodeSum'].apply(judge_empty)



print ('replace',round(time.clock()))
replaceText(weather)

#weather['PrecipTotal']=weather['PrecipTotal'].apply(precip)
weather['Date']=pd.to_datetime(weather['Date'])
weather.index=weather['Date']
weather1=weather[weather['Station']==1]
weather2=weather[weather['Station']==2]
print (' sort weather',round(time.clock()))
impute_missing_weather_station_values(weather1,weather2)



'''
title=["Date" ,"Tmax" ,"Tmin" ,"Tavg" ,
"DewPoint" , "WetBulb" ,"PrecipTotal" ,"Depart" , 
"ResultSpeed" ,"ResultDir" ,"AvgSpeed" ,"StnPressure", "SeaLevel" ,
'RA','BR','TSRA','TS','HZ','DZ','FG','VCTS','CODE_E']
'''


#===== begin here is my way =========

print (' assembling ',round(time.clock()))

X = assemble_X(train, weather1,weather2)
T = assemble_X(test, weather1,weather2)
y = assemble_y(train)
print (' assembling over',round(time.clock()))

scaler=preprocessing.StandardScaler().fit(X)

X1=scaler.transform(X)
T1=scaler.transform(T)


per_train=0.7

Z=np.hstack([X1,y])


Z=np.random.permutation(Z)
num_train=int(round(len(y)*per_train))

train_set=Z[0:num_train,0:-1]
train_lab=Z[0:num_train,-1]
test_set=Z[num_train: ,0:-1]
test_lab=Z[num_train: ,-1]
'''
clf=svm.SVC(C=4.0,gamma=100.0,class_weight={1:10})
clf.fit(train_set,train_lab)
p_lab=clf.predict(test_set)
'''
# grid_search way 
'''
clf = svm.SVC(C=2.0,gamma=100.0,class_weight={1:8})
print (' fitting ',round(time.clock()))
clf.fit(train_set,train_lab)
print (' predicting ',round(time.clock()))
p_lab=clf.predict(test_set)
'''
clf = RandomForestClassifier(n_estimators=350, max_depth=None,min_samples_split=1, random_state=0)

clf.fit(train_set,train_lab)
p_lab=clf.predict(test_set)
report(test_lab, p_lab)
p_lab2=clf.predict(T1)

sample['WnvPresent'] = p_lab2
sample.to_csv('RFtree.csv', index=False)
print (' sum of lab2 is ',sum(p_lab2))
