import numpy as np
import pandas as pd
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

#Filling nan
df_all = df_all.fillna(-1)

#####Feature engineering#######
#date_account_created
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]

#timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]

## dates
df_all['date_account_created'] = pd.to_datetime(df_all['date_account_created'])
df_all['date_first_active'] = pd.to_datetime((df_all.timestamp_first_active // 1000000), format='%Y%m%d')
df_all['date_first_booking'] = pd.to_datetime(df_all['date_first_booking'])

## dow
df_all['dac_dow'] = df_all['date_account_created'].dt.dayofweek
df_all['dfa_dow'] = df_all['date_first_active'].dt.dayofweek
df_all['dfb_dow'] = df_all['date_first_booking'].dt.dayofweek

## drops
df_all = df_all.drop(['date_account_created'], axis=1)
df_all = df_all.drop(['timestamp_first_active'], axis=1)
df_all = df_all.drop(['date_first_active'], axis=1)
df_all = df_all.drop(['id', 'date_first_booking'], axis=1)

#Age
av = df_all.age.values
df_all['age'] = np.where(np.logical_and(av>1919, av<1995), 2015-av, av)
df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)
    
print(df_all.shape)
print(list(df_all.columns.values))
print(df_all.head(5))

#Splitting train and test
vals = df_all.values
X = vals[:piv_train]
le = LabelEncoder()
y = le.fit_transform(labels)   
X_test = vals[piv_train:]

#Classifier
xgb = XGBClassifier(max_depth=5, learning_rate=0.2, n_estimators=50,
                    objective='multi:softprob', subsample=0.6, colsample_bytree=0.6, seed=0, silent=0)                  
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
sub.to_csv('sub2.csv',index=False)


  
    