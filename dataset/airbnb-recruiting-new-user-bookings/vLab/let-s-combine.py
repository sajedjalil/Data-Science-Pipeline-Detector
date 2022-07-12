import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier

np.random.seed(0)

def calculate_pct(device_type_time, total_time):
      return device_type_time/total_time if total_time > 0 else None
      

def get_user_device_type_time(fpath='../input/sessions.csv'):

    sessions = pd.read_csv(fpath)
    # sum up secs_elapsed time on each device_type for each user
    device_type_time = pd.pivot_table(sessions, index=['user_id'], columns=['device_type'], values='secs_elapsed', aggfunc=sum, fill_value=0)
    device_type_time.reset_index(inplace=True)
    # sum up elapsed time on all the devices for each user
    device_type_time['total_elapsed_time'] = device_type_time.sum(axis=1)
    
    # add attributes for usage percentage of each device type
    device_columns = device_type_time.columns[1:-2]  # exclude first column: user_id and last column: total_elapsed_time
    for column in device_columns:
        device_type_time[column+'_pct'] = device_type_time.apply(lambda row: calculate_pct(row[column], row['total_elapsed_time']), axis=1)
    
    
    print(device_type_time[device_type_time.total_elapsed_time > 0].head())

    return device_type_time
	

def merge_user_and_session_data(user_df, user_device_type_time_df=None):

    if not isinstance(user_device_type_time_df, pd.DataFrame):
        user_device_type_time_df = get_user_device_type_time()

    users_combined_df = pd.merge(user_df, user_device_type_time_df, left_on='id', right_on='user_id', how='left')
    return users_combined_df


#Loading data
df_train = pd.read_csv('../input/train_users_2.csv')
df_test = pd.read_csv('../input/test_users.csv')

user_device_type_time_df = get_user_device_type_time()
df_train = merge_user_and_session_data(df_train, user_device_type_time_df)
df_test = merge_user_and_session_data(df_test, user_device_type_time_df)

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

#Splitting train and test
vals = df_all.values
X = vals[:piv_train]
le = LabelEncoder()
y = le.fit_transform(labels)   
X_test = vals[piv_train:]

#Classifier
xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)                  
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

