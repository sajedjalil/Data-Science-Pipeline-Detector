import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import numpy as np


def get_season(x):
    summer=0
    fall=0
    winter=0
    spring=0
    if (x in [5, 6, 7]):
        summer=1
    if (x in [8, 9, 10]):
        fall=1
    if (x in [11, 0, 1]):
        winter=1
    if (x in [2, 3, 4]):
        spring=1
    return summer, fall, winter, spring


#Load Data with pandas, and parse the first column into datetime
train=pd.read_csv('../input/train.csv', parse_dates = ['Dates'])
test=pd.read_csv('../input/test.csv', parse_dates = ['Dates'])

#print 'Stage 1'

le_crime = preprocessing.LabelEncoder()
crime = le_crime.fit_transform(train.Category)

#Get binarized weekdays, districts, and hours.
days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = train.Dates.dt.hour
hour = pd.get_dummies(hour)
month = train.Dates.dt.month

#print 'Stage 2'

#Build new array
address_col = train.Address
add_col = []
for idx, temp in enumerate(address_col):
   if '/' in temp:
      add_col.append(0)
   else:
      add_col.append(1)
add_col = pd.DataFrame(np.array(add_col), columns=['add_col'])
#print add_col.head()
train_data = pd.concat([hour, days, district, add_col], axis=1)
hour = train.Dates.dt.hour
train_data['Awake'] = hour.apply(lambda x: 1 if (x==0 or (x>=8 and x<=23)) else 0)
train_data['Summer'], train_data['Fall'], train_data['Winter'], train_data['Spring']=zip(*month.apply(get_season))
train_data['crime']=crime

#Repeat for test data
days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)

hour = test.Dates.dt.hour
hour = pd.get_dummies(hour)
month = test.Dates.dt.month
address_col = test.Address
add_col = []
#add_col.append('Add')

for idx, temp in enumerate(address_col):
   if '/' in temp:
      add_col.append(0)
   else:
      add_col.append(1)
add_col = pd.DataFrame(np.array(add_col), columns=['add_col'])
test_data = pd.concat([hour, days, district, add_col], axis=1)
hour = test.Dates.dt.hour
test_data['Awake'] = hour.apply(lambda x: 1 if (x==0 or (x>=8 and x<=23)) else 0)
test_data['Summer'], test_data['Fall'], test_data['Winter'], test_data['Spring']=zip(*month.apply(get_season))


training, validation = train_test_split(train_data, train_size=.60)

#print 'Stage 3'

features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
 'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN', 'add_col', 'Summer', 'Fall', 'Winter', 'Spring', 'Awake']
features2 = [x for x in range(0,24)]
features = features + features2

training, validation = train_test_split(train_data, train_size=.60)
model = BernoulliNB()
model.fit(training[features], training['crime'])
#print 'predicting the model'
predicted = np.array(model.predict_proba(validation[features]))
log_loss(validation['crime'], predicted)

#Logistic Regression for comparison
model = LogisticRegression(C=.01)
model.fit(training[features], training['crime'])
predicted = np.array(model.predict_proba(validation[features]))
log_loss(validation['crime'], predicted)

#print 'predicting train model on test'
model = BernoulliNB()
model.fit(train_data[features], train_data['crime'])
predicted = model.predict_proba(test_data[features])

#Write results
result=pd.DataFrame(predicted, columns=le_crime.classes_)
result.to_csv('testResult.csv', index = True, index_label = 'Id' )