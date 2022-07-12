
import pandas as pd
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
import numpy as np
import gzip,csv
import time, datetime

train = pd.read_csv(r'../input/train.csv')
y= train.Category.values
train = train.drop(['Address','Category','Descript','Resolution'],axis=1)

month=list()
hour=list()
for i in train.Dates.values:
    t=time.strptime(i,"%Y-%m-%d %H:%M:%S")
    month.append(t[1])
    hour.append(t[3])    

train['Month']=pd.Series(month,index=train.index)
train['Hour']=pd.Series(hour,index=train.index)
train=train.drop(['Dates'],axis=1)

days = {}
cnt=0
for i in np.unique(train.DayOfWeek.values):
    days[i] = cnt
    cnt+=1

dict ={'DayOfWeek' : days}

PdDis = {}
cnt=0
for i in np.unique(train.PdDistrict.values):
    PdDis[i] = cnt
    cnt+=1

dict['PdDistrict'] = PdDis

train = train.replace(dict)

model = BernoulliNB()
#model = GaussianNB()
model.fit(train,y)

test = pd.read_csv(r'../input/test.csv')
idx = test.Id.values
test = test.drop(['Id','Address'],axis=1)
month=list()
hour=list()
for i in test.Dates.values:
    t=time.strptime(i,"%Y-%m-%d %H:%M:%S")
    month.append(t[1])
    hour.append(t[3])    

test['Month']=pd.Series(month,index=test.index)
test['Hour']=pd.Series(hour,index=test.index)
test=test.drop(['Dates'],axis=1)
test = test.replace(dict)

predicted = np.array(model.predict_proba(test))
labels = ['Id']
for i in model.classes_:
    labels.append(i)
with gzip.open('bernoulinb.csv.gz', 'wt') as outf:
  fo = csv.writer(outf, lineterminator='\n')
  fo.writerow(labels)

  for i, pred in enumerate(predicted):
    fo.writerow([i] + list(pred))

