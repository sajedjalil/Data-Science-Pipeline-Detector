# Based on Ben Hamner script from Springleaf
# https://www.kaggle.com/benhamner/springleaf-marketing-response/random-forest-example

import pandas as pd
import numpy as np

#load train and test
print("reading the train and test data\n")
train  = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')
store  = pd.read_csv('../input/store.csv')


#TODO covert r to python
#set na to mean
store.CompetitionOpenSinceMonth.fillna(7)
#set na to (mean+median)/2
store.CompetitionOpenSinceYear.fillna(2009)
train = pd.merge(train,store,on=['Store'],how='left')
test = pd.merge(test,store,on=['Store'],how='left')

# There are some NAs in the integer columns so conversion to zero
train=train.fillna(0)
test=test.fillna(0)

print("train data column names and details\n")
print(train.columns)
print(train)
print("test data column names and details\n")
print(test.columns)
print(test)

# looking at only stores that were open in the train set
# may change this later
train = train.loc[train.Open!=0,:]

# seperating out the elements of the date column for the train set
train['Date']=pd.to_datetime(train.Date)
train['month']=train['Date'].dt.month
train['year']=train['Date'].dt.year
train['day']=train['Date'].dt.day
#encode StateHoliday 
train.StateHoliday.loc[train.StateHoliday=='0']=0
train.StateHoliday.loc[train.StateHoliday=='a']=1
train.StateHoliday.loc[train.StateHoliday=='b']=2
train.StateHoliday.loc[train.StateHoliday=='c']=3
train.StateHoliday=train.StateHoliday.astype(float)

# removing the date column (since elements are extracted)
train = train.drop('Date', 1)


# seperating out the elements of the date column for the test set
test['Date']=pd.to_datetime(test.Date)
test['month']=test['Date'].dt.month
test['year']=test['Date'].dt.year
test['day']=test['Date'].dt.day
#encode StateHoliday 
test.StateHoliday.loc[test.StateHoliday=='0']=0
test.StateHoliday.loc[test.StateHoliday=='a']=1
test.StateHoliday.loc[test.StateHoliday=='b']=2
test.StateHoliday.loc[test.StateHoliday=='c']=3
test.StateHoliday=test.StateHoliday.astype(float)
# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
test = test.drop('Date', 1)
Id = test.Id
test = test.drop('Id', 1)
feature_name=[0,1]
feature_name.extend(range(5,20))
feature_name=train.columns[feature_name]
print("Feature Names\n")
print(feature_name)

print("assuming text variables are categorical & replacing them with numeric ids\n")
for f in feature_name:
  if train[f].dtype=="object" or train[f].dtype=="str" :
    train[f]= pd.factorize(train[f])[0]
    test[f]=pd.factorize(test[f])[0]


print("train data column names after slight feature engineering\n")
print(train.columns)
print("test data column names after slight feature engineering\n")
print(test.columns)

from sklearn.ensemble import RandomForestRegressor
clf=RandomForestRegressor(n_estimators=100)
clf.verbose = True
clf.n_jobs = 8

print("Random Forest Setup")
print(clf)
clf.fit(train[feature_name],np.log(train.Sales+1.0))

print("Predicting Sales\n")

pred = np.exp(clf.predict(test[feature_name]))-1.0
print(pred)
print("saving the submission file\n")
d = {'Id': Id.values, 'Sales': pred}
output = pd.DataFrame(data=d)
print(output)
output.to_csv('rf3.csv',index=False)
