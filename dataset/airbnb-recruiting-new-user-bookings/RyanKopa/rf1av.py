#https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import train_test_split


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

#####Feature engineering#######
#date_account_created
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x:
    list(map(int, x.split('-')))).values)
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
df_all = df_all.drop(['date_account_created'], axis=1)

#timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x:
    list(map(int, [x[:4],x[4:6],x[6:8],
    x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
df_all = df_all.drop(['timestamp_first_active'], axis=1)

#Age
av = df_all.age.values
df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

ohe_feats = ['gender', 'signup_method', 'signup_flow',
    'language', 'affiliate_channel', 'affiliate_provider',
    'first_affiliate_tracked', 'signup_app', 'first_device_type',
    'first_browser']

for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)


vals = df_all.values
X = vals[:piv_train]
X_test = vals[piv_train:]
le = LabelEncoder()
y = le.fit_transform(labels)
# xgb classifier
xgb = XGBClassifier(max_depth=6,
                    learning_rate=0.2,
                    n_estimators=30,
                    objective='multi:softprob',
                    subsample=0.5,
                    colsample_bytree=0.5,
                    seed=0)
eval_set  = [(X,y)]
num_trees = 600
xgb.fit(X, y, eval_set=eval_set,
    eval_metric = 'merror', early_stopping_rounds = 50)
yprob = xgb.predict_proba(X_test)
# RandomForestClassifier
rf1 = RandomForestClassifier(n_estimators=500, n_jobs=-1,
    criterion="entropy",
      oob_score = True, class_weight = "balanced_subsample",
      max_depth=10, max_features=6, min_samples_leaf=2, random_state=1)
rf1.fit(X, y)
rf1yprob = rf1.predict_proba(X_test)
# average probabilities
avgProb = yprob*(0.5)+rf1yprob*(0.5)

#Generate submission
idD = []
cts = []
for i in range(len(id_test)):
    idx = id_test[i]
    idD += [idx]*5
    cts += le.inverse_transform(
        np.argsort(avgProb[i])[::-1])[:5].tolist()

sub = pd.DataFrame(np.column_stack((idD, cts)),
    columns=['id', 'country'])
sub.to_csv('sub1.csv',index=False)