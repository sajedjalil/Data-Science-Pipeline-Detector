
# coding: utf-8


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
import xgboost

sessions= pd.read_csv('../input/sessions.csv')
sessions.action.fillna('nan',inplace=True)
sessions.action_type.fillna('nan',inplace=True)
sessions.action_detail.fillna('nan',inplace=True)
sessions.secs_elapsed.fillna(0,inplace=True)
#sessions['action_detail_2'] = sessions.action + '-' + sessions.action_type

#####session bigrams -> get user's flow semantics
#sessions['prev_details'] = sessions.action_detail.shift(1)
#sessions['prev_user'] = sessions.user_id.shift(1)
#sessions['action_detail_3'] = sessions['action_detail']+'-'+sessions['prev_details']
#sessions.ix[sessions.prev_user != sessions.user_id, 'action_detail_3'] = map(lambda x:x+'-begin', sessions[sessions.prev_user != sessions.user_id]['action_detail'].values)

sessions['new_session'] = map(lambda x:x if x > 60*15 else 0.0, sessions.secs_elapsed)

def flatten_df(df,delimeter=" ",suffix=""):
    df.columns = [suffix+delimeter.join(col).strip() for col in df.columns.values]
    return df

secs = sessions.groupby(['user_id'])['secs_elapsed'].aggregate({'sec_sum':np.sum, 'sec_median':np.median, 'sec_mean':np.mean, 'sec_max':np.max},fill_value=0).reset_index()

action_detail_gp = flatten_df(pd.pivot_table(sessions, index = ['user_id'],columns = ['action_detail'],values = 'secs_elapsed',aggfunc=[np.sum,np.median,np.max,np.mean],fill_value=0).reset_index())

action_type_gp = flatten_df(pd.pivot_table(sessions, index = ['user_id'],columns = ['action_type'],values = 'secs_elapsed',aggfunc=[np.sum,np.median,np.max,np.mean],fill_value=0).reset_index())

action_gp = flatten_df(pd.pivot_table(sessions, index = ['user_id'],columns = ['action'],values = 'secs_elapsed',aggfunc=[np.sum,np.median,np.max,np.mean],fill_value=0).reset_index())

device_type = flatten_df(pd.pivot_table(sessions, index = ['user_id'],columns = ['device_type'],values = 'secs_elapsed',aggfunc=[np.sum,np.median,np.max,np.mean],fill_value=0).reset_index())

sess = sessions.groupby(['user_id'])['new_session'].aggregate({'sec_sum':np.sum, 'sec_median':np.median, 'sec_mean':np.mean, 'sess_nz':np.count_nonzero, 'sess_len':len}).reset_index()

#old_action_type = flatten_df(pd.pivot_table(sessions, index = ['user_id'],columns = ['action_detail_3'],values = 'secs_elapsed',aggfunc=[np.sum,np.median,np.max,np.mean],fill_value=0).reset_index())

sessions_data_temp1 = pd.merge(action_detail_gp,device_type,on='user_id',how='inner',suffixes=('_action_detail_gp', '_device'))
sessions_data_temp2 = pd.merge(sessions_data_temp1,action_type_gp,on='user_id',how='inner',suffixes=('', '_action_type_gp'))
sessions_data_temp3 = pd.merge(sessions_data_temp2,action_gp,on='user_id',how='inner',suffixes=('', '_action_gp'))
sessions_data_temp4 = pd.merge(sessions_data_temp3,secs,on='user_id',how='inner',suffixes=('', '_secs'))
#sessions_data_temp5 = pd.merge(sessions_data_temp4,old_action_type,on='user_id',how='inner',suffixes=('', '_oaction'))
sessions_data = pd.merge(sessions_data_temp4,sess,on='user_id',how='inner',suffixes=('', '_sess'))

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
df_all = df_all.drop(['date_first_booking'], axis=1)
df_all['user_id'] = df_all.id.values
df_all = df_all.drop(['id'], axis=1)

#Filling nan
df_all = df_all.fillna(-1)

#date_account_created
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
df_all = df_all.drop(['date_account_created'], axis=1)

#timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
df_all = df_all.drop(['timestamp_first_active'], axis=1)

#Age
av = df_all.age.values
df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

my_data = pd.merge(df_all,sessions_data,on='user_id',how='left')
#my_data.tail()

#Splitting train and test
my_data.drop(['user_id'], axis=1, inplace=True)
X_train = my_data[:piv_train] 
X_test = my_data[piv_train:]

le = LabelEncoder()
y = le.fit_transform(labels)
X = X_train.values

dtrain = xgboost.DMatrix(X, label=y, missing=float('NaN'))

param = {'max_depth':12, 'eta':0.3, 'silent':1, 'objective':'multi:softprob', 'subsample':0.8, 'colsample_bytree':0.6, 'num_class':12, 'missing':float('NaN') }
param['nthread'] = -1
param['eval_metric'] = 'mlogloss'
evallist  = [(dtrain,'train')]
bst = xgboost.train(param.items(), dtrain, 30, evallist )

X_t = X_test.values

dtest = xgboost.DMatrix(X_t,missing=float('NaN'))
ypred = bst.predict(dtest)

#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(ypred[i])[::-1])[:5].tolist()

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])


sub.to_csv('sub.csv',index=False)


