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

df_transactions = pd.read_csv('../input/kkbox-churn-prediction-challenge/transactions_v2.csv')
df_transactions.head()
df_transactions.info()
df_transactions['transaction_date'] = df_transactions.transaction_date.apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN" )
df_transactions['membership_expire_date'] = df_transactions.membership_expire_date.apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN" )
transactions = df_transactions.groupby('msno').apply(lambda x: x.sort_values(by=['membership_expire_date'], ascending=False))

#cancelled_user = df_transactions.groupby('msno')['is_cancel'].sum().reset_index()
#cancelled_user.columns = ['msno', 'num_churn']
#cancelled_user.head()
#transactions_log = transactions.merge(cancelled_user, how='inner', on='msno')
#transactions_log.head()

#cancelled_user = df_transactions.groupby('msno').filter( lambda x: x['is_cancel'].sum() > 0)
#len(cancelled_user)
#for name, group in cancelled_user:
#    sort = transactions.get_group(name).sort_values(by=['membership_expire_date'])

#Testing print groupby values    
#df_transactions[(df_transactions.msno == '+0KcMm8JNCW08lTp3Lyz5Ger/47u3yj9H2xLf8lyAj8=')]
#transactions.get_group('+/w1UrZwyka4C9oNH3+Q8fUf3fD8R3EwWrx57ODIsqk=').sort_values(by=['membership_expire_date'])
#(transactions.get_group('+0KcMm8JNCW08lTp3Lyz5Ger/47u3yj9H2xLf8lyAj8=')['is_cancel'].sum() > 0)

#================ Finish cleaning transaction logs =============================

df_members = pd.read_csv('../input/kkbox-churn-prediction-challenge/members_v3.csv')
#remove bd due to outliners
del df_members['bd']

#convert gender to int value
gender = {'male':1, 'female':2}
df_members['gender'] = df_members['gender'].map(gender)

#fill data with na to "NAN"
#df_members['gender'] = df_members.gender.apply(lambda x: (x=="female" and 1) or (x=="male" and 0) or -1)
#df_members['city'] = df_members.city.apply(lambda x: int(x) if pd.notnull(x) else "NAN")
#df_members['registered_via'] = df_members.registered_via.apply(lambda x: int(x) if pd.notnull(x) else "NAN")
#change time format to YYYY-MM-DD if na fill in "NAN"
#df_members['registration_init_time'] = df_members.registration_init_time.apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN" )
# ==================== Finish cleaning members data ===========================

user_log =  pd.read_csv('../input/kkbox-churn-prediction-challenge/user_logs_v2.csv')
del user_log['date']
#group by msno
#print (str(np.shape(user_log)) + " -- Size of data large due to repeated msno")
counts = user_log.groupby('msno')['total_secs'].count().reset_index()
counts.columns = ['msno', 'days_listened']
sums = user_log.groupby('msno').sum().reset_index()
user_log = sums.merge(counts, how='inner', on='msno')
print (str(np.shape(user_log)) + " -- New size of data matches unique member count")
#find avg seconds played per song
user_log['secs_per_song'] = user_log['total_secs'].div(user_log['num_25']+user_log['num_50']+user_log['num_75']+user_log['num_985']+user_log['num_100'])
#==================== Finish cleaning user logs ===============================

#merge members and user logs and transaction logs
user_data = df_members.merge(user_log, on='msno', how='inner')
#user_data = user_data.merge(transactions_log, on='msno', how='inner')

df_members.shape
user_log.shape
user_data.shape

df_members.head()
user_log.head()
user_data.head()

#read by chunk
#for chunk in pd.read_csv(churn_data_path + 'user_logs.csv', chunksize=500000):
#    merged = members.merge(chunk, on='msno', how='inner')
#    user_data = pd.concat([user_data, merged])

#user_data.to_csv('user_logs2.csv', index=False)


#remove time < 0
#for col in user_data.columns[1:]:
#    outlier_count = user_data['msno'][user_data[col] < 0].count()
#    print (str(outlier_count) + " outliers in column " + col)
#user_data = user_data[user_data['total_secs'] >= 0]
#print (user_data['msno'][user_data['total_secs'] < 0].count())

#with train data and features correlation
df_train = pd.read_csv('../input/kkbox-churn/user_label_201703.csv', dtype={'is_churn': 'int8'})

corrmat = user_data[user_data.columns[1:]].corr()
f, ax = plt.subplots(figsize=(12, 9))
sb.heatmap(corrmat, vmax=1, cbar=True, annot=True, square=True);
plt.show()

#remove highly corelated inputs
#del user_data['num_100']
#del user_data['payment_plan_days']

#normalize data
#from sklearn.preprocessing import StandardScaler

#cols = user_data.drop(['msno','registration_init_time','gender'],axis=1).columns
#log_user_data = user_data.copy()
#log_user_data[cols] = np.log1p(user_data[cols])
#ss = StandardScaler()
#log_user_data[cols] = ss.fit_transform(user_data[cols])

#for col in cols:
 #   plt.figure(figsize=(15,7))
 #   plt.subplot(1,2,1)
 #   sb.distplot(user_data[col])
 #   plt.subplot(1,2,2)
 #   sb.distplot(log_user_data[col])
 #   plt.figure()

#training = df_train.merge(log_user_data, how='left', on='msno')
training = df_train.merge(user_data, how='left', on='msno')

#make all nan fill to non-na
#from sklearn.preprocessing import Imputer
#im = Imputer()
#train2 = im.fit_transform(training.drop(['msno','registration_init_time'],axis=1)))
#training = pd.DataFrame(data=train2)
#================================== need find how to make that array back to training, FOR NOW JUST DROP ANY NA IN ORDER TO RUN MODEL W/O ERROR==========================
training = training.fillna(-1)
training.info()
#training.head()
#X = training.drop(['is_churn','msno','registration_init_time', 'transaction_date','membership_expire_date'],axis=1).values
cols = [c for c in training.columns if c not in ['is_churn','msno']]
X = training[cols]
Y = training['is_churn']
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size=validation_size, random_state = seed)

lgb_params = {
    'learning_rate': 0.05,
    'application': 'binary',
    'max_depth': 5,
    'num_leaves': 128,
    'verbosity': -1,
    'metric': 'binary_logloss'
}
d_train = lgb.Dataset(X_train, label=Y_train)
d_valid = lgb.Dataset(X_validation, label=Y_validation)
watchlist = [d_train, d_valid]

model = lgb.train(lgb_params, train_set=d_train, num_boost_round=240, valid_sets=watchlist, early_stopping_rounds=50, verbose_eval=10) 


#scoring = 'accuracy'
#models = []
#models.append(('LR', LogisticRegression(random_state=seed)))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('RFC',RandomForestClassifier(random_state=seed)))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
# evaluate each model in turn
#results = []
#names = []
#for name, model in models:
#	kfold = model_selection.KFold(n_splits=10, random_state=seed)
#	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#	results.append(cv_results)
#	names.append(name)
#	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#	print(msg)

#use LDA model train then predict
#lda = LinearDiscriminantAnalysis()
#lda = lda.fit(X_train, Y_train)
#predictions = lda.predict(X_validation)
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))

#print prediction probability
#print(lda.predict_proba(X_validation))
#print(predictions)

#use voting ensemble
#eclf = VotingClassifier(estimators=models, voting='soft')
#eclf = eclf.fit(X_train, Y_train)
#predictions = eclf.predict(X_validation)
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
#print(predictions)

#predict TEST DATA
df_test = pd.read_csv('../input/kkbox-churn-prediction-challenge/sample_submission_v2.csv')
test_data = df_test.merge(user_data, how='left', on='msno')
test_data=test_data.fillna(-1)
test_data.info()


#X_test = test_data.drop(['is_churn','msno','registration_init_time','transaction_date','membership_expire_date'],axis=1).values
#X_test = test_data[cols]

lgb_pred = model.predict(test_data[cols])

test_data['is_churn'] = lgb_pred.clip(0.+1e-15, 1-1e-15)
test_data[['msno','is_churn']].to_csv('lgb_result.csv', index=False)

test_data
#predictions = eclf.predict_proba(X_test)
#print(eclf.predict_proba(X_test))
#print(Counter(predictions))
#result = pd.DataFrame({'msno': df_test.msno, 'is_churn': predictions[:,0]})

#save prediction data
#result.to_csv('result_final.csv', index=False)