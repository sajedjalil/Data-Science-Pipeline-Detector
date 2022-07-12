# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os, gc
print(os.listdir("../input"))

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb

print('Importing data...')
train = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')
prev = pd.read_csv('../input/previous_application.csv')
buro = pd.read_csv('../input/bureau.csv')
buro_balance = pd.read_csv('../input/bureau_balance.csv')
credit_card  = pd.read_csv('../input/credit_card_balance.csv')
POS_CASH  = pd.read_csv('../input/POS_CASH_balance.csv')
payments = pd.read_csv('../input/installments_payments.csv')


#Separate target variable
y = train['TARGET']
del train['TARGET']

df = pd.concat([train,test])

#Pre-processing buro_balance
print('Pre-processing buro_balance...')
buro_grouped_size = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].size()
buro_grouped_max = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max()
buro_grouped_min = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].min()

buro_counts = buro_balance.groupby('SK_ID_BUREAU')['STATUS'].value_counts(normalize = False)
buro_counts_unstacked = buro_counts.unstack('STATUS')
buro_counts_unstacked.columns = ['STATUS_0', 'STATUS_1','STATUS_2','STATUS_3','STATUS_4','STATUS_5','STATUS_C','STATUS_X',]
buro_counts_unstacked['MONTHS_COUNT'] = buro_grouped_size
buro_counts_unstacked['MONTHS_MIN'] = buro_grouped_min
buro_counts_unstacked['MONTHS_MAX'] = buro_grouped_max

buro = buro.join(buro_counts_unstacked, how='left', on='SK_ID_BUREAU')
del(buro_balance, buro_counts, buro_counts_unstacked, buro_grouped_max, buro_grouped_min, buro_grouped_size)
gc.collect()

#Pre-processing previous_application
print('Pre-processing previous_application...')
#One-hot encoding of categorical features in previous application data set
prev_cat_features = [pcol for pcol in prev.columns if prev[pcol].dtype == 'object']
prev = pd.get_dummies(prev, columns=prev_cat_features)
avg_prev = prev.groupby('SK_ID_CURR').mean()
cnt_prev = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
avg_prev['nb_app'] = cnt_prev['SK_ID_PREV']
del avg_prev['SK_ID_PREV']
del(prev, cnt_prev)
gc.collect()

#Pre-processing buro
print('Pre-processing buro...')
#One-hot encoding of categorical features in buro data set
buro_cat_features = [bcol for bcol in buro.columns if buro[bcol].dtype == 'object']
buro = pd.get_dummies(buro, columns=buro_cat_features)
avg_buro = buro.groupby('SK_ID_CURR').mean()
avg_buro['buro_count'] = buro[['SK_ID_BUREAU', 'SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
del avg_buro['SK_ID_BUREAU']
del(buro)
gc.collect()

#Pre-processing POS_CASH
print('Pre-processing POS_CASH...')
le = LabelEncoder()
POS_CASH['NAME_CONTRACT_STATUS'] = le.fit_transform(POS_CASH['NAME_CONTRACT_STATUS'].astype(str))
nunique_status = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
nunique_status2 = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
POS_CASH['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
POS_CASH['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
POS_CASH.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

#Pre-processing credit_card
print('Pre-processing credit_card...')
credit_card['NAME_CONTRACT_STATUS'] = le.fit_transform(credit_card['NAME_CONTRACT_STATUS'].astype(str))
nunique_status = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
nunique_status2 = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
credit_card['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
credit_card['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
credit_card.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

#Pre-processing payments
print('Pre-processing payments...')
avg_payments = payments.groupby('SK_ID_CURR').mean()
avg_payments2 = payments.groupby('SK_ID_CURR').max()
avg_payments3 = payments.groupby('SK_ID_CURR').min()
del avg_payments['SK_ID_PREV']

#Join data bases
print('Joining databases...')
df= df.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')

df = df.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

df = df.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')

df = df.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')

df = df.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')

df = df.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')

df = df.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')
del(avg_prev, avg_buro, avg_payments, avg_payments2, avg_payments3, credit_card, POS_CASH)

#########################
#One hot encoding the categoical Columns
##########################
categorical_columns = [col for col in df.columns if df[col].dtype == 'object']

df['NAME_TYPE_SUITE'].fillna('Unaccompanied', inplace=True)
df['OCCUPATION_TYPE'].fillna('Unemplyed', inplace=True)
df['FONDKAPREMONT_MODE'].fillna('not specified', inplace=True)
df['HOUSETYPE_MODE'].fillna('not specified', inplace=True)
df['WALLSMATERIAL_MODE'].fillna('Others', inplace=True)
df['EMERGENCYSTATE_MODE'].fillna('No', inplace=True)
#Label encoding
le=LabelEncoder()

for col in categorical_columns:
    data=df[col]
    le.fit(data.values)
    df[col]=le.transform(df[col])

del(data) 
gc.collect()   
#Onehotencoding
      
#train_scale=scale(train)
ohe=OneHotEncoder()

# Below ohe conversion is looking for 2d array. Need to convert pandas series to 2d array
for col in categorical_columns:
    X=ohe.fit_transform(df[col].values.reshape(-1,1)).toarray()
    dfOneHot=pd.DataFrame(X, columns=[(col+"_"+str(i)) for i in range(X.shape[1])])
    df=pd.concat([df, dfOneHot], axis=1)
    
dropped_cols=categorical_columns
df.drop(dropped_cols, axis=1, inplace=True)

# Seggregate the train and test data
data = df.iloc[:train.shape[0],:]
test = df.iloc[train.shape[0]:,]

del(df)
del(train)
gc.collect()
#Delete customer Id
del data['SK_ID_CURR']
del test['SK_ID_CURR']

#Create train and validation set
train_x, valid_x, train_y, valid_y = train_test_split(data, y, test_size=0.2, shuffle=True)

#------------------------Build LightGBM Model-----------------------
train_data=lgb.Dataset(train_x,label=train_y)
valid_data=lgb.Dataset(valid_x,label=valid_y)

#Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          'n_estimators' : '10000',
          'max_depth' : 8,
          'objective': 'binary',
          'nthread': 4,
          'num_leaves': 32,
          'learning_rate': 0.02,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 0.8715623,
          'subsample_freq': 1,
          'colsample_bytree': 0.9497036,
          'reg_alpha': 0.041545473,
          'reg_lambda': 0.073529,
          'min_split_gain': 0.0222415,
          'min_child_weight': 39.3259775,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'auc'
          }

#Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 2500,
                 valid_sets=valid_data,
                 early_stopping_rounds= 40,
                 verbose_eval= 100
                 )

print('Best AUC Score of LightGBM: {}'.format(lgbm.best_score['valid_0']['auc']))

#Predict on test set and write to submit
predictions_lgbm_prob = lgbm.predict(test)
submission=pd.read_csv("../input/sample_submission.csv", header=0)
submission.TARGET = predictions_lgbm_prob

submission.to_csv('lgbm_submit02.csv', index=False)