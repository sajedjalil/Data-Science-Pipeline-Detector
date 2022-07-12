__author__ = 'sonu'

import pandas as pd
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
import numpy as np
import gzip,csv

def fetchHour(a):
    return float(pd.to_datetime(a).hour)

train = pd.read_csv(r'../input/train.csv')
y= train.Category.values
dates = train.Dates
train = train.drop(['Address','Category','Dates','Descript','X','Y','Resolution'],axis=1)

train['Hour'] = dates.apply(fetchHour)
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
dates = test.Dates
test = test.drop(['Id','Dates','Address','X','Y'],axis=1)
test['Hour'] = dates.apply(fetchHour)
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
