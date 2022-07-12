import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
print(os.listdir('../input'))
file = ['train_users.csv', 'age_gender_bkts.csv', 'sessions.csv', 'countries.csv', 'test_users.csv']
data = {}
### Get data ###
for f in file:
    data[f.replace('.csv','')]=pd.read_csv('../input/'+f)
    #print(f)
    #print(data[f.replace('.csv','')].columns)
    
### Variables ###
train = data['train_users']
test = data['test_users']
age = data['age_gender_bkts']

sessions = data['sessions']
sessions['id'] = sessions['user_id']
sessions = sessions.drop(['user_id'],axis=1)
country = data['countries']
target = train['country_destination']
train = train.drop(['country_destination'],axis=1)

del data
##### Merging #####
print(len(train))
train1 = pd.merge(train,age,on='gender')
test1 = pd.merge(test,age,on='gender')
train2 = pd.merge(train,sessions,on='id')
test2 = pd.merge(test,sessions,on='id')
print(len(train1))
id = train['id']
train = train.drop(['id'],axis=1)
train = train.fillna(0)
test = test.fillna(0)
id_test = test['id']
print('size of id:\t',len(id_test))
test = test.drop(['id'],axis=1)
labelencoder={}
def labelencoderCustom(data,labelencoder={}):
    result = []
    for i in data:
        if i not in labelencoder:
            labelencoder[i] = len(labelencoder)+1
            result.append(labelencoder[i])
        else:
            result.append(labelencoder[i])
    return result,labelencoder
print('Type of columns')
for col in train.columns:
    if train[col].dtypes == object :
        #train = train.drop([col],axis=1)
        #le = LabelEncoder()
        #print(set(train[col]))
        #le.fit(list(train[col].values)+list(test[col].values))
        
        train[col],labelencoder = labelencoderCustom(train[col].values,labelencoder)
        test[col],labelencoder =  labelencoderCustom(test[col].values,labelencoder)
print('Columns:')
print(train.dtypes)
print('Columns test:')
print(test.dtypes)
train.to_csv('clean_train.csv',index=False)
test.to_csv('clean_test.csv', index=False)
target.to_csv('clean_target.csv', index=False)