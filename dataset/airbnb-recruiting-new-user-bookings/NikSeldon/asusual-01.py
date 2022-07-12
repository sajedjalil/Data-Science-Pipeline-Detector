# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier

np.random.seed(0)

#Loading data
df_train = pd.read_csv('../input/train_users_2.csv')
df_test = pd.read_csv('../input/test_users.csv')
labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0]

#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
#Removing id and date_first_booking
df_all = df_all.drop(['id', 'date_first_booking'], axis=1)
#Filling nan
df_all = df_all.fillna(-1)

import datetime
df_all['date_account_created']=pd.to_datetime(df_all['date_account_created'], format= "%Y-%m-%d")
df_all['dac_year']=df_all.date_account_created.dt.year
df_all['dac_month']=df_all.date_account_created.dt.month
df_all['dac_week']=df_all.date_account_created.dt.week
df_all['dac_day']=df_all.date_account_created.dt.day
df_all['dac_wday']=df_all.date_account_created.dt.dayofweek
df_all = df_all.drop(['date_account_created'], axis=1)

#timestamp_first_active
tfa = pd.DataFrame(np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(str, [x[:4],x[4:6],x[6:8]]))).values))
df_all["tfa"] = tfa[0].map(str) +"-"+ tfa[1].map(str)+"-"+tfa[2].map(str)
df_all['tfa']=pd.to_datetime(df_all['tfa'], format= "%Y-%m-%d")
df_all['tfa_year']=df_all.tfa.dt.year
df_all['tfa_month']=df_all.tfa.dt.month
df_all['tfa_week']=df_all.tfa.dt.week
df_all['tfa_day']=df_all.tfa.dt.day
df_all['tfa_wday']=df_all.tfa.dt.dayofweek
df_all = df_all.drop(['timestamp_first_active'], axis=1)
df_all = df_all.drop(['tfa'], axis=1)

#Age
av = df_all.age.values
df_all['age'] = np.where((av>=1900), 2015-av, av)
df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)
    
    #Splitting train and test
vals = df_all.values
X = vals[:piv_train]
le = LabelEncoder()
y = le.fit_transform(labels)   
X_test = vals[piv_train:]

#Classifier
xgb = XGBClassifier(max_depth=6, learning_rate=0.25, n_estimators=42,
                    objective='multi:softprob', subsample=0.6, colsample_bytree=0.6, seed=0)                  
xgb.fit(X, y)
y_pred = xgb.predict_proba(X_test) 

#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()
    
    #Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('sub.csv',index=False)