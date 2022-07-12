import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


print("read train")
train = pd.read_csv('../input/train.csv')
print("read sub")
test = pd.read_csv('../input/sample_submission_zero.csv')
print("read members")
mem = pd.read_csv('../input/members.csv')

#print("read trans")
#transactions = pd.read_csv('../input/transactions.csv',nrows=1000)

test = test.merge(mem[['msno', 'registered_via']], on='msno', how='left')
train = train.merge(mem[['msno', 'registered_via']], on='msno', how='left')
train.registered_via.fillna(-1)
test.registered_via.fillna(-1)

print("read logs")
user_logs = pd.read_csv('../input/user_logs.csv',nrows=1000000)
user_logs.info()
print(user_logs.head())

#Split the log data in train/test
train_logs = user_logs.loc[(user_logs.date>=20170101) & (user_logs.date<=20170130) ]
test_logs  = user_logs.loc[(user_logs.date>=20170201) & (user_logs.date<=20170228) ]

#Calculating the log of minutes of music per user
print("group by")
md_train =  train_logs[['msno', 'total_secs']].groupby(['msno'], as_index=False).median()
md_train['total_minutes'] = np.log((md_train.total_secs)).astype('int') 
md_train.info()

md_test =  test_logs[['msno', 'total_secs']].groupby(['msno'], as_index=False).median()
md_test['total_minutes'] = np.log(md_test.total_secs).astype('int') 
md_test.info()


print("merge")
print(train.columns)
print(md_train.columns)

#Joining previous calculations
train = train.merge(md_train, how='left')
train.total_minutes= train.total_minutes.fillna(0)
print(train.info())

test = test.merge(md_test, how='left')
test.total_minutes= test.total_minutes.fillna(0)
print(test.info())


print("churn mean")
#total churn mean (for exceptions)
base_mean = train.is_churn.mean()

#churn mean per minutes log
churn_mean = train[['registered_via', 'total_minutes','is_churn']].groupby(['registered_via', 'total_minutes'], as_index=False).mean()
churn_mean.columns = ['registered_via', 'total_minutes','churn_mean']
print(churn_mean.info())

# members = pd.read_csv('../input/members.csv')
test = test.merge(churn_mean, how='left')
test.churn_mean = test.churn_mean.fillna(base_mean)
print(test.info())

print(train.shape)
print(test.shape)
# print(transactions.shape)
# print(user_logs.shape)
# print(members.shape)

test['is_churn'] = test.churn_mean

test[['msno','is_churn']].to_csv('subm.csv', index=False)

