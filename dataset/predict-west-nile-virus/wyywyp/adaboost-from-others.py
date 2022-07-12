
import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing, cross_validation
from sklearn import svm ,grid_search
import sklearn.metrics as metrics

# Load dataset 
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sampleSubmission.csv')
weather = pd.read_csv('../input/weather.csv')

# Get labels
labels = train.WnvPresent.values

# Not using codesum for this benchmark
weather = weather.drop('CodeSum', axis=1)

# Split station 1 and 2 and join horizontally
weather_stn1 = weather[weather['Station']==1]
weather_stn2 = weather[weather['Station']==2]
weather_stn1 = weather_stn1.drop('Station', axis=1)
weather_stn2 = weather_stn2.drop('Station', axis=1)
weather = weather_stn1.merge(weather_stn2, on='Date')

# replace some missing values and T with -1
weather = weather.replace('M', -1)
weather = weather.replace('-', -1)
weather = weather.replace('T', -1)
weather = weather.replace(' T', -1)
weather = weather.replace('  T', -1)

# Functions to extract month and day from dataset
# You can also use parse_dates of Pandas.
def create_month(x):
    return x.split('-')[1]

def create_day(x):
    return x.split('-')[2]


def report(test_lab,p_lab):

	print ('sum of p_lab :',sum(p_lab))
	print ('classification_report :' )

	print (metrics.classification_report(test_lab,p_lab))

	print ('the f1_score :',metrics.f1_score(test_lab,p_lab))

	print ('AUC_score :',metrics.roc_auc_score(test_lab,p_lab))



train['month'] = train.Date.apply(create_month)
train['day'] = train.Date.apply(create_day)
test['month'] = test.Date.apply(create_month)
test['day'] = test.Date.apply(create_day)

# Add integer latitude/longitude columns
train['Lat_int'] = train.Latitude.apply(int)
train['Long_int'] = train.Longitude.apply(int)
test['Lat_int'] = test.Latitude.apply(int)
test['Long_int'] = test.Longitude.apply(int)

# drop address columns
train = train.drop(['Address', 'AddressNumberAndStreet','WnvPresent', 'NumMosquitos'], axis = 1)
test = test.drop(['Id', 'Address', 'AddressNumberAndStreet'], axis = 1)

# Merge with weather data
train = train.merge(weather, on='Date')
test = test.merge(weather, on='Date')
train = train.drop(['Date'], axis = 1)
test = test.drop(['Date'], axis = 1)

# Convert categorical data to numbers
lbl = preprocessing.LabelEncoder()
lbl.fit(list(train['Species'].values) + list(test['Species'].values))
train['Species'] = lbl.transform(train['Species'].values)
test['Species'] = lbl.transform(test['Species'].values)

lbl.fit(list(train['Street'].values) + list(test['Street'].values))
train['Street'] = lbl.transform(train['Street'].values)
test['Street'] = lbl.transform(test['Street'].values)

lbl.fit(list(train['Trap'].values) + list(test['Trap'].values))
train['Trap'] = lbl.transform(train['Trap'].values)
test['Trap'] = lbl.transform(test['Trap'].values)

# drop columns with -1s
train = train.ix[:,(train != -1).any(axis=0)]
test = test.ix[:,(test != -1).any(axis=0)]

#cross validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, labels, test_size=0.3, random_state=0)


# Random Forest Classifier 
#clf = ensemble.AdaBoostClassifier(n_estimators=1000)
clf=svm.SVC(C=1,gamma=800)
clf.fit(X_train, y_train)
p_lab=clf.predict(X_test)
print(report(y_test,p_lab))

# create predictions and submission file
#predictions = clf.predict_proba(test)[:,1]
#sample['WnvPresent'] = predictions
#sample.to_csv('adaboost.csv', index=False)


