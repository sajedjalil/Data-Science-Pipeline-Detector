# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb
import lightgbm as lgb
from collections import Counter
from datetime import datetime
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Input data files are available in the "../input/" directory.
 
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df_train = pd.read_csv('../input/kkbox-churn/user_label_201703.csv', dtype={'is_churn': 'int8'})

df_test = pd.read_csv('../input/kkbox-churn-prediction-challenge/sample_submission_v2.csv')

df_transactions1 = pd.read_csv('../input/mytransaction/user_churn_hist1.csv')
df_transactions2 = pd.read_csv('../input/mytransaction/user_churn_hist2.csv')
df_transactions3 = pd.read_csv('../input/mytransaction/user_churn_hist3.csv')
df_transactions4 = pd.read_csv('../input/mytransaction/user_churn_hist4.csv')
df_transactions5 = pd.read_csv('../input/mytransaction/user_churn_hist5.csv')
df_transactions6 = pd.read_csv('../input/mytransaction/user_churn_hist6.csv')
df_transactions7 = pd.read_csv('../input/mytransaction/user_churn_hist7.csv')
df_transactions8 = pd.read_csv('../input/mytransaction/user_churn_hist8.csv')
df_transactions9 = pd.read_csv('../input/mytransaction/user_churn_hist9.csv')
df_transactions10 = pd.read_csv('../input/mytransaction/user_churn_hist10.csv')
df_transactions11 = pd.read_csv('../input/mytransaction/user_churn_hist11.csv')
df_transactions12 = pd.read_csv('../input/mytransaction/user_churn_hist12.csv')

df_transactions = df_transactions1.append(df_transactions2, ignore_index=True)
df_transactions = df_transactions.append(df_transactions3, ignore_index=True)
df_transactions = df_transactions.append(df_transactions4, ignore_index=True)
df_transactions = df_transactions.append(df_transactions5, ignore_index=True)
df_transactions = df_transactions.append(df_transactions6, ignore_index=True)
df_transactions = df_transactions.append(df_transactions7, ignore_index=True)
df_transactions = df_transactions.append(df_transactions8, ignore_index=True)
df_transactions = df_transactions.append(df_transactions9, ignore_index=True)
df_transactions = df_transactions.append(df_transactions10, ignore_index=True)
df_transactions = df_transactions.append(df_transactions11, ignore_index=True)
df_transactions = df_transactions.append(df_transactions12, ignore_index=True)

training = df_train.merge(df_transactions, how='left', on='msno')
test_data = df_test.merge(df_transactions, how='left', on='msno')
#================ Finish cleaning transaction logs =============================

user_log1 =  pd.read_csv('../input/kkbox-churn-prediction-challenge/user_logs.csv', nrows=22000000)
user_log =  pd.read_csv('../input/kkbox-churn-prediction-challenge/user_logs_v2.csv')
user_log = user_log.append(user_log1, ignore_index=True)
del user_log['date']
#group by msno
counts = user_log.groupby('msno')['total_secs'].count().reset_index()
counts.columns = ['msno', 'days_listened']
sums = user_log.groupby('msno').sum().reset_index()
user_log = sums.merge(counts, how='inner', on='msno')
print (str(np.shape(user_log)) + " -- New size of data matches unique member count")
#find avg seconds played per song
user_log['secs_per_song'] = user_log['total_secs'].div(user_log['num_25']+user_log['num_50']+user_log['num_75']+user_log['num_985']+user_log['num_100'])

training = df_train.merge(user_log, how='left', on='msno')
test_data = df_test.merge(user_log, how='left', on='msno')
#==================== Finish cleaning user logs ===============================

df_members = pd.read_csv('../input/kkbox-churn-prediction-challenge/members_v3.csv')
#remove bd due to outliners
del df_members['bd']

#convert gender to int value
gender = {'male':1, 'female':2}
df_members['gender'] = df_members['gender'].map(gender)

#get number days from 31 mar 17 to reg init date
current = datetime.strptime('20170331', "%Y%m%d").date()
df_members['num_days'] = df_members.registration_init_time.apply(lambda x: (current - datetime.strptime(str(int(x)), "%Y%m%d").date()).days if pd.notnull(x) else "NAN" )
del df_members['registration_init_time']

training = training.merge(df_members, how='left', on='msno')
test_data = test_data.merge(df_members, how='left', on='msno')
# ==================== Finish cleaning members data ===========================

training = training.fillna(-1)
test_data=test_data.fillna(-1)

corrmat = training[training.columns[1:]].corr()
f, ax = plt.subplots(figsize=(12, 9))
sb.heatmap(corrmat, vmax=1, cbar=True, annot=True, square=True);
plt.show()

#remove highly corelated inputs
del training['total_secs']
del test_data['total_secs']
#del user_data['payment_plan_days']

cols = [c for c in training.columns if c not in ['is_churn','msno']]
X = training[cols]
Y = training['is_churn']
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size=validation_size, random_state = seed)


lgb_params = {
    'learning_rate': 0.05,
    'application': 'binary',
    'max_depth': 40,
    'num_leaves': 2000,
    'verbosity': -1,
    'metric': 'binary_logloss'
}
d_train = lgb.Dataset(X_train, label=Y_train)
d_valid = lgb.Dataset(X_validation, label=Y_validation)
watchlist = [d_train, d_valid]

model = lgb.train(lgb_params, train_set=d_train, num_boost_round=1000, valid_sets=watchlist, early_stopping_rounds=50, verbose_eval=10) 

lgb_pred = model.predict(test_data[cols])

test_data['is_churn'] = lgb_pred.clip(0.+1e-15, 1-1e-15)
test_data[['msno','is_churn']].to_csv('lgb_result.csv', index=False)