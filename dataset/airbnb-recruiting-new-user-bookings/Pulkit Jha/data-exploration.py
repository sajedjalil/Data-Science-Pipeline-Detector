
import pandas as pd
import os
import numpy as np

#print(os.listdir('../input'))
file = ['train_users.csv', 'age_gender_bkts.csv', 'sessions.csv', 'countries.csv', 'test_users.csv']
data = {}

for f in file:
    data[f.replace('.csv','')]=pd.read_csv('../input/'+f)
    
train = data['train_users']
test = data['test_users']
age = data['age_gender_bkts']
sessions = data['sessions']
country = data['countries']

#print(age.head(100))

for data in [train,test,age,sessions,country]:
    
    print("Top View of data...\n")
    print(data.head())
    print(data.info())
    print(data.apply(lambda x: x.nunique(),axis=0))

#keep all positive diff. as 1 and negative as 0 in version 2
from datetime import datetime

#raw_data['Mycol'] =  pd.to_datetime(raw_data['Mycol'], format='%d%b%Y:%H:%M:%S.%f')
#train
train['ac'] = pd.to_datetime(train['date_account_created'])
#train['dfb'] = pd.to_datetime(train['date_first_booking'])
train['fa'] = pd.to_datetime(train['timestamp_first_active'],format='%Y%m%d%H%M%S')

train = train.drop(['date_account_created','date_first_booking','timestamp_first_active'],axis=1)

train['year_ac'] = train['ac'].apply(lambda x : x.year)
train['month_ac'] = train['ac'].apply(lambda x : x.month)
train['day_ac'] = train['ac'].apply(lambda x : x.day)

#train['year_fb'] = train['dfb'].apply(lambda x : x.year)
#train['month_fb'] = train['dfb'].apply(lambda x : x.month)
#train['day_fb'] = train['dfb'].apply(lambda x : x.day)

train['today'] = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
train['lifetime'] = (train['today']- train['ac'])/np.timedelta64(1, 'D')
train['recentivity']= ( train['today'] - train['fa'])/np.timedelta64(1, 'D')

#test
test['ac'] = pd.to_datetime(test['date_account_created'])
#test['dfb'] = pd.to_datetime(test['date_first_booking'])
test['fa'] = pd.to_datetime(test['timestamp_first_active'],format='%Y%m%d%H%M%S')

test['year_ac'] = test['ac'].apply(lambda x : x.year)
test['month_ac'] = test['ac'].apply(lambda x : x.month)
test['day_ac'] = test['ac'].apply(lambda x : x.day)

test['today'] = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
test['lifetime'] = (test['today']- test['ac'])/np.timedelta64(1, 'D')
test['recentivity']= ( test['today'] - test['fa'])/np.timedelta64(1, 'D')

test = test.drop(['date_account_created','date_first_booking','timestamp_first_active'],axis=1)

#test['year_fb'] = test['dfb'].apply(lambda x : x.year)
#test['month_fb'] = test['dfb'].apply(lambda x : x.month)
#test['day_fb'] = test['dfb'].apply(lambda x : x.day)

#add other gender to unknown
#print(pd.crosstab(train.gender,train.country_destination).reset_index())
train['gender'][train['gender']=='OTHER']='-unknown-'
test['gender'][test['gender']=='OTHER']='-unknown-'
#print(pd.crosstab(train.gender,train.country_destination).reset_index())

#age
#print(pd.crosstab(train.age,train.country_destination).reset_index())
train.loc[train.age<15,'age']=15
train.loc[train.age>100,'age']=100

# get average, std, and number of NaN values in airbnb_df
average_age   = train["age"].mean()
std_age      = train["age"].std()
count_nan_age = train["age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(average_age - std_age, average_age + std_age, size = count_nan_age)
train["age"][np.isnan(train["age"])] = rand_1

test.loc[test.age<15,'age']=15
test.loc[test.age>100,'age']=100

# get average, std, and number of NaN values in test_df
average_age_test   = test["age"].mean()
std_age_test       = test["age"].std()
count_nan_age_test = test["age"].isnull().sum()

rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)
test["age"][np.isnan(test["age"])]     = rand_2

train['age'] = train['age'].astype(int)
test['age']   = test['age'].astype(int)

#print(pd.crosstab(train.age,train.country_destination).reset_index())
train['age_range'] = pd.cut(train["age"], [0, 20, 40, 60, 80, 100])
test['age_range'] = pd.cut(test["age"], [0, 20, 40, 60, 80, 100])

#signup method
#print(pd.crosstab(train.signup_method,train.country_destination).reset_index())

#signup flow
#print(pd.crosstab(train.signup_flow,train.country_destination).reset_index())
#train['sf'] = train.replace({'signup_flow' : {1:999,4:999,5:999,6:999,8:999,10:999,15:999,16:999,20:999,21:999}})
train['sf'] = train['signup_flow'].replace([1,4,5,6,8,10,15,16,20,21],999)
test['sf'] = test['signup_flow'].replace([1,4,5,6,8,10,15,16,20,21],999)
#print(pd.crosstab(train.sf,train.country_destination).reset_index())

#language
#print(pd.crosstab(train.language,train.country_destination).reset_index())
lang = list(np.unique(train['language']))
lang = [x for x in lang if x!='en']
train['lg'] = train['language'].replace(lang,'other') 

lang = list(np.unique(test['language']))
lang = [x for x in lang if x!='en']
test['lg'] = test['language'].replace(lang,'other') 

#print(pd.crosstab(train.lg,train.country_destination).reset_index())

#affiliate channel
#print(pd.crosstab(train.affiliate_channel,train.country_destination).reset_index())
train['ac'] = train['affiliate_channel'].replace(['api','content','remarketing'],'other')
test['ac'] = test['affiliate_channel'].replace(['api','content','remarketing'],'other')
#print(pd.crosstab(train.ac,train.country_destination).reset_index())

#affiliate provider
#print(pd.crosstab(train.affiliate_provider,train.country_destination).reset_index())
train['ac'] = train['affiliate_provider'].replace(['baidu','email-marketing','facebook-open-graph','gsp','meetup','naver','wayn','yandex'],'other')
test['ac'] = test['affiliate_provider'].replace(['baidu','email-marketing','facebook-open-graph','gsp','meetup','naver','wayn','yandex'],'other')
#print(pd.crosstab(train.ac,train.country_destination).reset_index())

#first_affiliate_tracked 
#print(pd.crosstab(train.first_affiliate_tracked,train.country_destination).reset_index())

count_first_affiliate = 7    # len(np.unique(airbnb_df["first_affiliate_tracked"].value_counts()))

count_nan_department = train["first_affiliate_tracked"].isnull().sum()
count_nan_department_test   = test["first_affiliate_tracked"].isnull().sum()

rand_1 = np.random.randint(0, count_first_affiliate, size = count_nan_department)
rand_2 = np.random.randint(0, count_first_affiliate, size = count_nan_department_test)

range_departments = train['first_affiliate_tracked'].value_counts().index
range_departments_test   = test['first_affiliate_tracked'].value_counts().index

train["first_affiliate_tracked"][train["first_affiliate_tracked"] != train["first_affiliate_tracked"]] = range_departments[rand_1]
test["first_affiliate_tracked"][test["first_affiliate_tracked"] != test["first_affiliate_tracked"]]       = range_departments_test[rand_2]
train['fat'] = train['first_affiliate_tracked'].replace(['local ops','marketing','product'],'tracked-other')
test['fat'] = test['first_affiliate_tracked'].replace(['local ops','marketing','product'],'tracked-other')
#print(pd.crosstab(train.fat,train.country_destination).reset_index())

print(train.info())

from sklearn import preprocessing

for f in train.columns:
    if f == "country_destination" or f == "id": continue
    if train[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(train[f].values) + list(test[f].values)))
        train[f] = lbl.transform(list(train[f].values))
        test[f]   = lbl.transform(list(test[f].values))
        
print(train.head())
print (test.head())


import sys
#sys.exit()
X_train = train.drop(["country_destination",'id','today','age_range','fa'],axis=1)
Y_train = train["country_destination"]
X_test  = test.drop(["id",'today','age_range','fa'],axis=1).copy()
print(train.info())

country_num_dic = {'NDF': 0, 'US': 1, 'other': 2, 'FR': 3, 'IT': 4, 'GB': 5, 'ES': 6, 'CA': 7, 'DE': 8, 'NL': 9, 'AU': 10, 'PT': 11}
num_country_dic = {y:x for x,y in country_num_dic.items()}

Y_train    = Y_train.map(country_num_dic)

# Random Forests
#from sklearn.ensemble import RandomForestClassifier

#random_forest = RandomForestClassifier(n_estimators=100)
#random_forest.fit(X_train, Y_train)
#Y_pred = random_forest.predict(X_test)
#random_forest.score(X_train, Y_train)

# Xgboost 
import xgboost as xgb
params = {"objective": "multi:softmax", "num_class": 12}

T_train_xgb = xgb.DMatrix(X_train, Y_train)
X_test_xgb  = xgb.DMatrix(X_test)

gbm = xgb.train(params, T_train_xgb, 20)
Y_pred = gbm.predict(X_test_xgb)

# convert type to integer
Y_pred = Y_pred.astype(int)

# change values back to original country symbols
Y_pred = pd.Series(Y_pred).map(num_country_dic)

# Create submission

country_df = pd.DataFrame({
        "id": test["id"],
        "country": Y_pred
    })

submission = pd.DataFrame(columns=["id", "country"])

# sort countries according to most probable destination country 
for key in country_df['country'].value_counts().index:
    print(key)
    submission = pd.concat([submission, country_df[country_df["country"] == key]], ignore_index=True)

submission.to_csv('airbnb.csv', index=False)