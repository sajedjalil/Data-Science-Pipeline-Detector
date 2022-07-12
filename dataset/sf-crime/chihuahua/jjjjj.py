import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import numpy as np
import zipfile
 
#Load Data with pandas, and parse the first column into datetime
z = zipfile.ZipFile('../input/train.csv.zip')
y = zipfile.ZipFile('../input/test.csv.zip')
train = pd.read_csv(z.open('train.csv'), parse_dates = ['Dates'])
test=pd.read_csv(y.open('test.csv'), parse_dates = ['Dates'])

#Convert crime labels to numbers
le_crime = preprocessing.LabelEncoder()
crime = le_crime.fit_transform(train.Category)
 
#Get binarized weekdays, districts, and hours.
days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = train.Dates.dt.hour
hour = pd.get_dummies(hour) 
 
#Build new array
train_data = pd.concat([hour, days, district], axis=1)
train_data['crime']=crime
 
#Repeat for test data
days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)
 
hour = test.Dates.dt.hour
hour = pd.get_dummies(hour) 
 
test_data = pd.concat([hour, days, district], axis=1)
 
training, validation = train_test_split(train_data, train_size=.60)

features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
 'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
 
training, validation = train_test_split(train_data, train_size=.60)
model = BernoulliNB()
model.fit(training[features], training['crime'])
predicted = np.array(model.predict_proba(validation[features]))
log_loss(validation['crime'], predicted) 
 
#Logistic Regression for comparison
model = LogisticRegression(C=.01)
model.fit(training[features], training['crime'])
predicted = np.array(model.predict_proba(validation[features]))
log_loss(validation['crime'], predicted) 

model = BernoulliNB()
model.fit(train_data[features], train_data['crime'])
predicted = model.predict_proba(test_data[features])

#Write results
result=pd.DataFrame(predicted, columns=le_crime.classes_)