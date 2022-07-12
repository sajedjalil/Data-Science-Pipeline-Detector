#加入了时间特征


from __future__ import division
import string
import numpy as np
from numpy.random import randn
from pandas import Series, DataFrame
import pandas as pd
import csv
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import time
import datetime
from sklearn.ensemble import GradientBoostingClassifier
#%matplotlib inline

        
file = open("../input/train.csv")
fout = open('subset_datatrain.csv','w')
n = 0
fout.write(file.readline())
for line in file:
    arr = line.strip().split(',')
    is_book = int(arr[-6])
    if is_book == 1:
        fout.write(line)
fout.close()
file.close()



dategroup = ['2013Jan','2013May','2013Sep','2014Jan','2014May','2014Sep','2015Jan','2015May','2015Sep','2016Jan','2016May','2016Sep']
chingroup = ['2013Jan','2013May','2013Sep','2014Jan','2014May','2014Sep','2015Jan','2015May','2015Sep','2016Jan','2016May','2016Sep']
dateix = [[] for i in range(12)]
chinix = [[] for i in range(12)]

def datedeal(date):
    n = len(date)
    for i in range(n):
        a = date[i]
        if type(a) == type(0.1):
            a = '2015-01-01'
        if int(a[1])>0 or int(a[2])>1 or int(a[2])<1 or int(a[3])>6 or int(a[3])<3:   #大于2016或小于2013的年份全部换成2015
            date[i] = '2015-01-01'
    return pd.to_datetime(date)

def frameDateDeal(frame, datename):
    frame[datename]=frame[datename].fillna('2015-01-01')
    dateix = [[] for i in range(12)]
    datevalue = datedeal(frame[datename].values)
    datevalue = Series(np.arange(len(datevalue)),index = datevalue)
    for i in range(48):
        y = divmod(i,12)[0]
        r = divmod(i,12)[1]
        n = divmod(i,4)[0]
        if r<9:
            dateix[n].extend(datevalue['201'+str(3+y)+'-0'+str(r+1)].values)
        else:
            dateix[n].extend(datevalue['201'+str(3+y)+'-'+str(r+1)].values)
    for i in range(12):
        frame[datename].values[dateix[i]] = i
    return frame


featurelist = ['user_id','user_location_city','srch_destination_id','hotel_market','srch_ci']
whlist = ['user_id','user_location_city','srch_destination_id','hotel_market','srch_ci','hotel_cluster']
trainpart = pd.read_csv('subset_datatrain.csv',na_values=['--  '],usecols = whlist)
os.remove('subset_datatrain.csv')
trainpart = frameDateDeal(trainpart,'srch_ci')
RFdata = trainpart[featurelist].values
RFpara = {'data':RFdata,'feature_names':featurelist,'target':trainpart['hotel_cluster'].values,
'target_names':np.arange(100)}

testpart = pd.read_csv('../input/test.csv',na_values=['--  '],usecols = featurelist)
# for i in range(9):
#     testpart['srch_ci'].values[chinix[i]] = i
testpart = frameDateDeal(testpart,'srch_ci')
testdata = testpart[featurelist].values


from sklearn.ensemble import RandomForestClassifier
RFclf = RandomForestClassifier(n_estimators=30,
    max_depth=18, random_state=0).fit(RFdata, RFpara['target'])
print('RFclf OK!')

now = datetime.datetime.now()
path = 'submission_RF_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
out = open(path, "w")
out.write("id,hotel_cluster\n")

import random
mn = divmod(len(testdata),40000)
m = mn[0]
n = mn[1]
eventid = 0
for i in range(m+1):
    clus = []
    if i<m:
        a = RFclf.predict_proba(testdata[(i*40000):(i+1)*40000,:])
        b=np.argsort(a)[:,-10:]
        for ind in b:
            clus = []
            for ix in ind:
                clus.append(str(ix))
            clus = random.sample(clus,5)
            out.write(str(eventid)+","+" ".join(clus)+"\n")
            eventid += 1
    else:
        a = RFclf.predict_proba(testdata[(i*40000):len(testdata),:])
        b=np.argsort(a)[:,-10:]
        for ind in b:
            clus = []
            for ix in ind:
                clus.append(str(ix))
            clus = random.sample(clus,5)            
            out.write(str(eventid)+","+" ".join(clus)+"\n")
            eventid += 1
