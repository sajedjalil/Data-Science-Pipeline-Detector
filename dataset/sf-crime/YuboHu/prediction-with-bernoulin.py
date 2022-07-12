import math
import pandas as pd
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.linear_model import LogisticRegression
import numpy as np
import gzip,csv
import time, datetime

train = pd.read_csv(r'../input/train.csv')

#train=train[800000:len(train)]
y= train.Category.values
train = train.drop(['Address','Category','Descript','Resolution'],axis=1)

month=list()
hour=list()
year=list()
day=list()
for i in train.Dates.values:
    t=time.strptime(i,"%Y-%m-%d %H:%M:%S")
    month.append(t[1])
    hour.append(t[3]) 
    year.append(t[0])
    day.append(t[2])


train['Month']=pd.Series(month,index=train.index)
train['Hour']=pd.Series(hour,index=train.index)
train['Day']=pd.Series(day,index=train.index)
train['Year']=pd.Series(year,index=train.index)
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
#df = pd.DataFrame(train, columns = ['DayOfWeek', 'PdDistrict', 'Year', 'Month', 'Hour','X','Y','Category'])
#df.to_csv("train.csv")#, sep='\t', encoding='utf-8')


model = BernoulliNB()
#model = GaussianNB()
#model=MultinomialNB()
#model=LogisticRegression()
model.fit(train,y)


predicted = np.array(model.predict_proba(train))
labels = ['Id']
for i in model.classes_:
    labels.append(i)
with gzip.open('bernoulinb.csv.gz', 'wt') as outf:
  fo = csv.writer(outf, lineterminator='\n')
  fo.writerow(labels)
    
  for i, pred in enumerate(predicted):
    fo.writerow([i] + list(pred))

Classes={}
cnt=1
for i in model.classes_:
	Classes[i]=cnt
	cnt+=1
	
predicted['Category']=pd.Series(y,index=train.index)	
logloss=0
cnt=0
for i, pred in enumerate(predicted):
    logloss+=math.log(list(pred)[Classes[y[i]]-1]+1e-15)
    if list(pred).index(max(list(pred)))==Classes[y[i]]-1:
        cnt+=1


logloss/=-len(predicted) 
accuracy=cnt/len(predicted)
print(logloss)
print(accuracy)