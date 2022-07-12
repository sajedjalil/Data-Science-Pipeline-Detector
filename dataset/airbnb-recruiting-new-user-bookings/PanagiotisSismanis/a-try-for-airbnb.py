# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import sys

from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import preprocessing

import xgboost as xgb

from time import time

pd.set_option('display.max_rows',9999)

seed = 999
t0 = time()

d1 = pd.read_csv('../input/train_users_2.csv', parse_dates=[1,3])

print(d1.head())

y = d1['country_destination']
del d1['country_destination']

d1['year_account_created'] = d1['date_account_created'].dt.year
d1['month_account_created'] = d1['date_account_created'].dt.month
d1['wkday_account_created'] = d1['date_account_created'].dt.dayofweek

del d1['date_account_created']

tot_features = d1.columns

missing_data_list = []

for v in tot_features:
    n1 = d1[v].isnull().sum()
    if (n1 > 0):
        print(v, n1)
        missing_data_list.append(v)

print(missing_data_list)

unique_keys = []

for v in missing_data_list:
    w = d1[v].unique()
    unique_keys.append(w)
    
n1 = len(missing_data_list)

del d1['date_first_booking']  # test_users.csv is empty

d1['year_first_active'] = d1['timestamp_first_active'].apply(lambda x: int(str(x)[:4]))
d1['month_first_active'] = d1['timestamp_first_active'].apply(lambda x: int(str(x)[4:6]))
d1['wkday_first_active'] = d1['timestamp_first_active'].apply(lambda x: pd.Timestamp(str(x)[:8])).dt.dayofweek

d1['first_affiliate_tracked'].fillna('other',inplace=True)

age_avg = d1['age'].describe()[1]

d1['age'].fillna(age_avg, inplace=True)  # missing with mean

d1['age'] = np.where(d1['age'] < 15, age_avg, d1['age'])
d1['age'] = np.where(d1['age'] > 95, age_avg, d1['age'])

del d1['timestamp_first_active']

r = pd.Series(np.random.rand(d1.shape[0]))

d1['gender'] = np.where(d1['gender']=='OTHER', r.apply(lambda x: 'MALE' if x <= 0.5 else 'FEMALE'), d1['gender'])

d1['gender'] = np.where(d1['gender']=='-unknown-', r.apply(lambda x: 'MALE' if x <= 0.5 else 'FEMALE'), d1['gender'])

print('*' * 80)

print('reading test...')

d2 = pd.read_csv('../input/test_users.csv', parse_dates=[1,3])

print(d2.shape)

test_id_keep_sep = d2['id']

d2['year_account_created'] = d2['date_account_created'].dt.year
d2['month_account_created'] = d2['date_account_created'].dt.month
d2['wkday_account_created'] = d2['date_account_created'].dt.dayofweek

del d2['date_account_created']

tot_features = d2.columns

missing_data_list = []

for v in tot_features:
    n1 = d2[v].isnull().sum()
    if (n1 > 0):
        print(v, n1)
        missing_data_list.append(v)

print(missing_data_list)

unique_keys = []

for v in missing_data_list:
    w = d2[v].unique()
    unique_keys.append(w)
    
n1 = len(missing_data_list)

del d2['date_first_booking']

d2['year_first_active'] = d2['timestamp_first_active'].apply(lambda x: int(str(x)[:4]))
d2['month_first_active'] = d2['timestamp_first_active'].apply(lambda x: int(str(x)[4:6]))
d2['wkday_first_active'] = d2['timestamp_first_active'].apply(lambda x: pd.Timestamp(str(x)[:8])).dt.dayofweek

d2['first_affiliate_tracked'].fillna('other',inplace=True)

age_avg2 = d2['age'].describe()[1]

d2['age'].fillna(age_avg2, inplace=True)  # missing with mean

d2['age'] = np.where(d2['age'] < 15, age_avg2, d2['age'])
d2['age'] = np.where(d2['age'] > 95, age_avg2, d2['age'])

del d2['timestamp_first_active']

r = pd.Series(np.random.rand(d2.shape[0]))

d2['gender'] = np.where(d2['gender']=='OTHER', r.apply(lambda x: 'MALE' if x <= 0.5 else 'FEMALE'), d2['gender'])

d2['gender'] = np.where(d2['gender']=='-unknown-', r.apply(lambda x: 'MALE' if x <= 0.5 else 'FEMALE'), d2['gender'])

print('*' * 80)

print('reading...sessions')

d4 = pd.read_csv('../input/sessions.csv')

d4 = d4[ d4['user_id'].notnull() == True ]

d4['action'].fillna('other', inplace=True)
d4['action_type'].fillna('other', inplace=True)
d4['action_detail'].fillna('other', inplace=True)

zzz15 = d4['secs_elapsed'].describe()[1]

d4['secs_elapsed'].fillna(zzz15, inplace=True)

d4['id'] = d4['user_id']

del d4['user_id']

d1['country_destination'] = y

del y

print('Before merging...')

print(d1.head())
print(d2.head())

d3 = pd.merge(d1, d4, on='id')

d2 = pd.merge(d2, d4, on='id', how='left')  # don't miss data from test

print('After merging...')

print(d3.head())
print(d2.head())
quit()


d2['action'].fillna('other', inplace=True)
d2['action_type'].fillna('other', inplace=True)
d2['action_detail'].fillna('other', inplace=True)
d2['device_type'].fillna('other', inplace=True)

d2['secs_elapsed'].fillna(zzz15, inplace=True)

del d4

print('*' * 80)

d1['countryTdestination'] = d1['country_destination']

del d1['country_destination']

print("Converting objects into numbers...")

LE3 = preprocessing.LabelEncoder()
yy1 = pd.Series(d2['id'])
LE3.fit(list(yy1.values))
yy1 = LE3.transform(list(yy1.values))

object_features = d1.select_dtypes(include=['object']).columns

object_features1 = []

for v in object_features[:-1]:
  object_features1.append(v)

for v in object_features1:
    if v=='id':
      LE3 = preprocessing.LabelEncoder()
      LE3.fit( list(d1[v].values) + list(d2[v].values) )
      d1[v] = LE3.transform( list(d1[v].values) )
      d2[v] = LE3.transform( list(d2[v].values) )
      print('object',v,'completed...',sep=' ')
    else:
      LE1 = preprocessing.LabelEncoder()
      LE1.fit( list(d1[v].values) + list(d2[v].values) )
      d1[v] = LE1.transform( list(d1[v].values) )
      d2[v] = LE1.transform( list(d2[v].values) )
      d1[v] = d1[v].astype('float')
      d2[v] = d2[v].astype('float')
      print('object',v,'completed...',sep=' ')
    
print('*' * 80)

for v in ['countryTdestination']:
    LE2 = preprocessing.LabelEncoder()
    LE2.fit( list(d1[v].values) )
    d1[v] = LE2.transform( list(d1[v].values) )
#    d1[v] = d1[v].astype('float')
    print('object',v,'completed...',sep=' ')

print('*' * 80)

rng = np.random.RandomState(31337)

d1.rename(columns={'signup_method' : 'signupTmethod',
                   'signup_flow' : 'signupTflow',
                   'affiliate_channel' : 'affiliateTchannel',
                   'affiliate_provider' : 'affiliateTprovider',
                   'first_affiliate_tracked' : 'firstTaffiliateTtracked',
                   'signup_app' : 'signupTapp',
                   'first_device_type' : 'firstTdeviceTtype',
                   'first_browser' : 'firstTbrowser',
                   'year_account_created' : 'yearTaccountTcreated',
                   'month_account_created' : 'monthTaccountTcreated',
                   'wkday_account_created' : 'wkdayTaccountTcreated',
                   'year_first_active' : 'yearTfirstTactive',
                   'month_first_active' : 'monthTfirstTactive',
                   'wkday_first_active' : 'wkdayTfirstTactive',
                   'action_type' : 'actionTtype',
                   'action_detail' : 'actionTdetail',
                   'device_type' : 'deviceTtype',
                   'secs_elapsed' : 'secsTelapsed'}, inplace=True)

d2.rename(columns={'signup_method' : 'signupTmethod',
                   'signup_flow' : 'signupTflow',
                   'affiliate_channel' : 'affiliateTchannel',
                   'affiliate_provider' : 'affiliateTprovider',
                   'first_affiliate_tracked' : 'firstTaffiliateTtracked',
                   'signup_app' : 'signupTapp',
                   'first_device_type' : 'firstTdeviceTtype',
                   'first_browser' : 'firstTbrowser',
                   'year_account_created' : 'yearTaccountTcreated',
                   'month_account_created' : 'monthTaccountTcreated',
                   'wkday_account_created' : 'wkdayTaccountTcreated',
                   'year_first_active' : 'yearTfirstTactive',
                   'month_first_active' : 'monthTfirstTactive',
                   'wkday_first_active' : 'wkdayTfirstTactive',
                   'action_type' : 'actionTtype',
                   'action_detail' : 'actionTdetail',
                   'device_type' : 'deviceTtype',
                   'secs_elapsed' : 'secsTelapsed'}, inplace=True)
                   
                   
print('d1..columns',d1.columns,sep=' ')
print('d2..columns',d2.columns,sep=' ')

n10 = len(d1.columns)
nn1 = n10-1

xx = np.array(np.arange(1,n10-1),dtype='int64')

names1 = d1.columns[ xx ]

print(d1.head())

train1 = d1

test101 = d2

train1.data = d1[ xx ]
train1.target = d1[ [nn1] ]
train1.feature_names = names1

# split 90/10 train-calibration
#X_train, X_cal, y_train, y_cal = train_test_split(train1.data, train1.target, test_size=0.10, random_state=1)
X_train = d1[ xx ]
y_train = d1[ [nn1] ]

names = feature_names = names1

print(".......names")
print(names)
      
print('_' * 80)
    
print("Going for a model...")
    
print("Parallel Parameter optimization")
         
param_dist = {'objective':'multi:softprob',
              'n_estimators': 3000,
              'max_depth': 23,
              'eta': 0.1,
              'nthread': 4,
              'silent': 1,
              'num_class': 12}
    
 #   if __name__ == "__main__":
    
#y_train = np.array(y_train).ravel() 
#y_cal = np.array(y_cal).ravel() 

print(param_dist)

clf = xgb.XGBClassifier(param_dist)

print(y_train[:10],sep=' ')
        
clf.fit(X_train[names1], y_train)
        
print("Validating")
        
print("predicting test data...")

test_preds1 = clf.predict_proba(test101[names1])
        
f1 = pd.DataFrame(test_preds1, columns=['prob00','prob01','prob02','prob03',
                                       'prob04','prob05','prob06','prob07',
                                       'prob08','prob09','prob10','prob11',])
                                       
f1['id'] = test101['id']

# groupby id & aggregate!

g1 = f1.groupby('id')

g11 = g1.agg(np.mean).reset_index()

del g11['id']

g111 = np.array(g11)

#Taking the 5 classes with highest probabilities
ids11 = []  #list of id's
ctrs11 = []  #list of countries

for i in range(len(test_id_keep_sep)):
    index11 = test_id_keep_sep[i]
    ids11 += [index11] * 5
    ctrs11 += LE2.inverse_transform(np.argsort(g111[i])[::-1])[:5].tolist()

#Generate submission
sub11 = pd.DataFrame(np.column_stack((ids11, ctrs11)), columns=['id', 'country'])

f11 = pd.DataFrame({'id': sub11['id'], 'country': sub11['country']},columns=['id','country'])

f11.to_csv('draft_237cc_5.csv', index=False)


    
