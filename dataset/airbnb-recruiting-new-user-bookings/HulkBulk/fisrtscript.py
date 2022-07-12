import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
print(os.listdir('../input'))
file = ['train_users.csv', 'age_gender_bkts.csv', 'sessions.csv', 'countries.csv', 'test_users.csv']
data = {}
for f in file:
    data[f.replace('.csv','')]=pd.read_csv('../input/'+f)
    
train = data['train_users']
test = data['test_users']
train = train.fillna(-100)
test = test.fillna(-100)

age = data['age_gender_bkts']
sessions = data['sessions']
country = data['countries']
target = train['country_destination']
train = train.drop(['country_destination'],axis=1)
def dateParseYear(date):
    #2010-06-28
    #print(date)
    
    try:
        spl = date.split('-')
        return int(spl[0])
    except:
        return 0
        
def dateParseMonth(date):
    
    try:
        spl = date.split('-')
        return int(spl[1])
    except:
        return 0
def dateParseDay(date):
    
    try:
        spl = date.split('-')
        return int(spl[2])
    except:
        return 0
train['year_c'] = train['date_account_created'].apply(dateParseYear)
test['year_c'] = test['date_account_created'].apply(dateParseYear)

train['month_c'] = train['date_account_created'].apply(dateParseMonth)
test['month_c'] = test['date_account_created'].apply(dateParseMonth)

train['day_c'] = train['date_account_created'].apply(dateParseDay)
test['day_c'] = test['date_account_created'].apply(dateParseDay)
train = train.drop(['date_account_created'],axis=1)
test = test.drop(['date_account_created'],axis=1)

train['year_f'] = train['date_first_booking'].apply(dateParseYear)
test['year_f'] = test['date_first_booking'].apply(dateParseYear)

train['month_f'] = train['date_first_booking'].apply(dateParseMonth)
test['month_f'] = test['date_first_booking'].apply(dateParseMonth)

train['day_f'] = train['date_first_booking'].apply(dateParseDay)
test['day_f'] = test['date_first_booking'].apply(dateParseDay)
train = train.drop(['date_first_booking'],axis=1)
test = test.drop(['date_first_booking'],axis=1)
#first_device_type Windows Desktop/Mac Desktop/ iPhone
for col in train.columns:
    uniquevalues = set(train[col].values)
    print("Unique value:\t",len(uniquevalues))
    if len(uniquevalues) ==1:
        train = train.drop([col],axis=1)
        test = test.drop([col],axis=1)
    print(train[col])

id = train['id']
train = train.drop(['id'],axis=1)

id_test = test['id']
print('size od id:\t',len(id_test))
test = test.drop(['id'],axis=1)
labelencoder={}
def labelencoderCustom(data,col,labelencoder={}):
    if col not in labelencoder:
        labelencoder[col] = {}
    result = []
    for i in data:
        if i not in labelencoder[col]:
            labelencoder[col][i] = len(labelencoder[col])+1
            result.append(labelencoder[col][i])
        else:
            result.append(labelencoder[col][i])
    return result,labelencoder
print('Type of columns')
for col in train.columns:
    if train[col].dtypes == object :
        #train = train.drop([col],axis=1)
        #le = LabelEncoder()
        #print(set(train[col]))
        #le.fit(list(train[col].values)+list(test[col].values))
        train[col],labelencoder = labelencoderCustom(train[col].values,col,labelencoder)
        test[col],labelencoder =  labelencoderCustom(test[col].values,col,labelencoder)
print('Columns:')
print(train.dtypes)
print('Columns test:')
print(test.dtypes)
train = train.fillna(0)
test = test.fillna(0)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=600,criterion='gini')
rf = rf.fit(train,target)
sample_submission = {}
sample_submission['id'] = id_test
#print(len(test))
#print(len(sample_submission['id'].values))
#print(sample_submission.columns)
sample_submission['country'] = rf.predict(test)
s = pd.DataFrame.from_dict(sample_submission)
s.to_csv('sub.csv',index=False)
