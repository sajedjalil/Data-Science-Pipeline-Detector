# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
path = "../input/"
# Any results you write to the current directory are saved as output.
import gc
import numpy as np
import pandas as pd
import random

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']


dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        }
        
dtypes_test = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16'
        }

lines = 184903891 # total number of rows in the complete dataset
skiplines = np.random.choice(np.arange(1, lines), size=lines-1-15000000, replace=False)
print("reading data...")
#sort the list
skiplines=np.sort(skiplines)
train = pd.read_csv(path+'train.csv', skiprows=skiplines, dtype=dtypes, usecols=train_cols, header=0)
train_attributed = pd.DataFrame()
chunksize = 10 ** 6
#in each chunk, filter for values that have 'is_attributed'==1, and merge these values into one dataframe
for chunk in pd.read_csv(path+'train.csv', chunksize=chunksize, dtype=dtypes):
    filtered = (chunk[(np.where(chunk['is_attributed']==1, True, False))])
    train_attributed = pd.concat([train_attributed, filtered], ignore_index=True)
    
train = pd.concat([train[train.is_attributed ==0], train_attributed], ignore_index=True)
test = pd.read_csv(path+'test.csv', dtype=dtypes_test, usecols=test_cols, header=0,)
len_train = len(train)
train=train.append(test)

del test
gc.collect()
del skiplines
gc.collect()
del train_attributed
gc.collect()
del filtered
gc.collect()

print("done reading data...")

train['click_time'] = pd.to_datetime(train['click_time'])
#train['period']= pd.cut(train.click_time.dt.hour,[-1,6,12,18,24],labels=['Night','Morning','Afternoon','Evening'])
train['hour']= train.click_time.dt.hour
train['day'] = train.click_time.dt.day.astype('uint8')

train.drop('click_time', axis=1,inplace=True)
train.drop('attributed_time', axis=1,inplace=True)


channel_count = train.groupby(['ip','day','hour'])['channel'].count().reset_index()
channel_count.columns = ['ip','day','hour','channel_count']
train = pd.merge(train, channel_count, how='left', on=['ip','day','hour'])
train['channel_count'].fillna(0, inplace=True)
del channel_count
gc.collect()

channel_count = train.groupby(['ip','app'])['channel'].count().reset_index()
channel_count.columns = ['ip','app','ip_app']
train = pd.merge(train, channel_count, how='left', on=['ip','app'])
train['ip_app'].fillna(0, inplace=True)
del channel_count
gc.collect()

channel_count = train.groupby(['ip','app','os'])['channel'].count().reset_index()
channel_count.columns = ['ip','app','os','ip_app_os']
train = pd.merge(train, channel_count, how='left', on=['ip','app','os'])
train['ip_app_os'].fillna(0, inplace=True)
del channel_count
gc.collect()


#features = ['ip', 
#            'app',
#            'os',
#            'channel',
#            'device']

#for n in range(0,len(features)):
#    count = train.groupby(features[n])['channel'].count().reset_index()
#    count.columns = [features[n], features[n]+'_count']
#    train = pd.merge(train, count, how='left', on=[features[n]])
#    train[features[n]+'_count'].fillna(0, inplace=True)
    
    
#pair_features = ['os','channel','device']

#for n in range(0,3):

#    count = train.groupby(['app', pair_features[n]])['channel'].count().reset_index()
#    count.columns = ['app', pair_features[n], 'app_'+pair_features[n]+'_count']
#    train = pd.merge(train, count, how='left', on=['app', pair_features[n]])
#    train['app_'+pair_features[n]+'_count'].fillna(0, inplace=True)

    
    
    
    
from sklearn.model_selection import train_test_split


### Split the train and test ###
test = train[len_train:]
train = train[:len_train]


print("splitting the data...")
X_train, X_test, y_train, y_test, = train_test_split(train.drop(['is_attributed','ip'], axis=1), 
                                                     train['is_attributed'], 
                                                     test_size = .1, 
                                                     random_state=123)

x_train, x_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size = .1,
                                                  random_state=12)

X_submission_id = pd.read_csv(path+'test.csv', dtype='int', usecols=['click_id'])
X_submission = test.drop(['is_attributed'], axis=1) 
X_submission.drop(['ip'], axis=1, inplace=True)


del test
gc.collect()

del train
gc.collect()

##### SMOTE #####
#from imblearn.over_sampling import SMOTE
#sm = SMOTE(random_state=123)
#x_train_res, y_train_res = sm.fit_sample(x_train, y_train)

print("start training the model...")

##### XGBOOST #####
from xgboost import XGBClassifier 
eval_set = [(x_val, y_val)]
classifier_xgb = XGBClassifier(objective = "binary:logistic",
                tree_method = "hist",
                 eval_metric = "auc",
                 nthread = 8,
                 eta = 0.3,
                 grow_policy = "lossguide",
                 max_leaves=1400,
                 min_child_weight=0,
                 max_depth = 0,
                 subsample = 0.8,
                 colsample_bytree = 0.8,
                 scale_pos_weight = 100,
                 n_estimators = 50,
                 alpha=4)

#classifier_xgb.fit(x_train_res, y_train_res)
classifier_xgb.fit(x_train, y_train, eval_set=eval_set, verbose=True, early_stopping_rounds=10)

del x_train
gc.collect()
del y_train
gc.collect()
del x_val
gc.collect()
del y_val
gc.collect()
del X_test
gc.collect()
del y_test
gc.collect()
del eval_set
gc.collect()

#from sklearn.metrics import accuracy_score
#y_val_pred_xgb = classifier_xgb.predict(x_val)
#print('XGBoost accuracy score of the validation set:')
#print(accuracy_score(y_val, y_val_pred_xgb))


#y_pred_xgb = classifier_xgb.predict(X_test)
#print('XGBoost accuracy score of the test set:')
#print(accuracy_score(y_test, y_pred_xgb))

#from sklearn.metrics import classification_report
#report = classification_report(y_test, y_pred_xgb)
#print(report)


#generate prediction for the test set

X_submission_id['is_attributed'] = classifier_xgb.predict(X_submission)
del X_submission
gc.collect()
X_submission_id.to_csv('submission8.csv', float_format='%.8f', index = False)
