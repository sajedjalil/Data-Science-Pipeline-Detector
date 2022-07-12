import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from time import gmtime, strftime


np.random.seed(0)

#Loading data
df_train = pd.read_csv('../input/train_users_2.csv')
df_test = pd.read_csv('../input/test_users.csv')
labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
id_train = df_train['id']
id_test = df_test['id']
piv_train = df_train.shape[0]

# rename id to user_id
df_train.rename(columns={'id': 'user_id'},inplace=True)
df_test.rename(columns={'id': 'user_id'},inplace=True)
#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
# Filling nan
df_all = df_all.fillna(-1)

# Loading session data
df_sessions = pd.read_csv('../input/sessions.csv')
grpby = df_sessions.groupby(['user_id'])['secs_elapsed'].sum().reset_index()
grpby.columns = ['user_id','secs_elapsed']
action_type = pd.pivot_table(df_sessions, index = ['user_id'],columns = ['action_type'],values = 'action',aggfunc=len,fill_value=0).reset_index()
action_type = action_type.drop(['booking_response'],axis=1)
device_type = pd.pivot_table(df_sessions, index = ['user_id'],columns = ['device_type'],values = 'action',aggfunc=len,fill_value=0).reset_index()
device_type = device_type.drop(['Blackberry','Opera Phone','iPodtouch','Windows Phone'],axis=1)
sessions_data = pd.merge(action_type,device_type,on='user_id',how='inner')
sessions_data = pd.merge(sessions_data,grpby,on='user_id',how='inner')
sessions_data.drop(['secs_elapsed'],axis=1,inplace=True)

#####Feature engineering#######
#date_account_created
# dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
# df_all['dac_year'] = dac[:,0]
# df_all['dac_month'] = dac[:,1]
# df_all['dac_day'] = dac[:,2]
df_all = df_all.drop(['date_account_created'], axis=1)

#timestamp_first_active
# tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
# df_all['tfa_year'] = tfa[:,0]
# df_all['tfa_month'] = tfa[:,1]
# df_all['tfa_day'] = tfa[:,2]
df_all = df_all.drop(['timestamp_first_active'], axis=1)

# timestampe comparsion
# tc = np.zeros(len(df_all))
# for i in range(len(df_all)):
#     xx = datetime.datetime(tfa[i,0],tfa[i,1],tfa[i,2],0,0,0) - datetime.datetime(dac[i,0],dac[i,1],dac[i,2],0,0,0)
#     tc[i] = np.sign(xx.total_seconds())
# df_all['tc'] = tc

#Age
av = df_all.age.values
df_all['age'] = np.where(np.logical_or(av<14, av>90), -1, av)

#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

# merge with session data
ohe_session = list(sessions_data.columns)
df_all = pd.merge(df_all,sessions_data,on='user_id',how='left')
df_all = df_all.fillna(-1)
# Removing id and date_first_booking
df_all = df_all.drop(['user_id', 'date_first_booking'], axis=1)

#Splitting train and test
vals = df_all.values
X = vals[:piv_train]
le = LabelEncoder()
y = le.fit_transform(labels)
X_test = vals[piv_train:]

# Classifier 1
rf = RandomForestClassifier(n_estimators=1000,min_samples_leaf=50, max_features="auto", n_jobs=6, random_state=0)
rf.fit(X, y)
y_pred1 = rf.predict_proba(X_test)

# Looking at Feature Importance
# features = list(df_all.columns.values)
# importances = rf.feature_importances_
# indices = np.argsort(importances)
# ind=[]
# for i in indices:
#     ind.append(features[i])
# plt.figure(1)
# plt.title('Feature Importances')
# plt.barh(range(len(indices)), importances[indices], color='b', align='center')
# plt.yticks(range(len(indices)),ind)
# plt.xlabel('Relative Importance')
# plt.show()
# Taking the 5 classes with highest probabilities
ids1 = []  #list of ids
cts1 = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids1 += [idx] * 5
    y_pred1[i] /= sum(y_pred1[i])
    cts1 += le.inverse_transform(np.argsort(y_pred1[i])[::-1])[:5].tolist()
#Generate submission
# time_str = strftime("%Y-%m-%d %H:%M:%S", gmtime())
# sub = pd.DataFrame(np.column_stack((ids1, cts1)), columns=['id', 'country'])
# sub.to_csv(time_str+'RF_sub.csv',index=False)

# Classifier 2
xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=43,objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
xgb.fit(X, y)
y_pred2 = xgb.predict_proba(X_test)

#Taking the 5 classes with highest probabilities
ids2 = []  #list of ids
cts2 = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids2 += [idx] * 5
    y_pred2[i] /= sum(y_pred2[i])
    cts2 += le.inverse_transform(np.argsort(y_pred2[i])[::-1])[:5].tolist()
#Generate submission
# sub = pd.DataFrame(np.column_stack((ids2, cts2)), columns=['id', 'country'])
# sub.to_csv(time_str+'XGB_sub.csv',index=False)

##Generate submission Essemble
y_pred3 = 0.5 * y_pred1 + 0.5 * y_pred2
ids3 = []  #list of ids
cts3 = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids3 += [idx] * 5
    cts3 += le.inverse_transform(np.argsort(y_pred3[i])[::-1])[:5].tolist()
sub = pd.DataFrame(np.column_stack((ids3, cts3)), columns=['id', 'country'])
sub.to_csv('sub.csv',index=False)