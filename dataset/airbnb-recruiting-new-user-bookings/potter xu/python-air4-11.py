import pandas as pd
import numpy as np
sessions=pd.read_csv("../input/sessions.csv")
train=pd.read_csv("../input/train_users_2.csv")
print(train.head())
print(train.info())
train.age=train.age.apply(lambda e: 2016-e if e >1000 else e)

train.age.hist()
train.age=train.age.fillna(int(train.age.mean()))
                 
train["date_first_booking_m"]=train.date_first_booking.apply(lambda e :e  if pd.isnull(e) else str(e)[5:7])
train["date_first_booking_y"]=train.date_first_booking.apply(lambda e :e  if pd.isnull(e) else str(e)[0:4])
train["date_account_created_m"]=train.date_account_created.apply(lambda e :e  if pd.isnull(e) else str(e)[5:7])
train["date_account_created_y"]=train.date_account_created.apply(lambda e :e  if pd.isnull(e) else str(e)[0:4])

train.head()
# ###选择部分变量，建模（决策树、NB、LR模型）
train.age1=pd.cut(train.age,[0,20,40,60,80,100,120])
train_x=pd.concat([pd.get_dummies(train.gender, prefix='gender'),
               pd.get_dummies(train.signup_method, prefix='signup_method'),
                pd.get_dummies(train.language,prefix='language'),
                   pd.get_dummies(train.affiliate_channel,prefix='affiliate_channel'),
                   pd.get_dummies(train.affiliate_provider,prefix='affiliate_provider'),
               pd.get_dummies(train.first_affiliate_tracked, prefix='first_affiliate_tracked'),
                   pd.get_dummies(train.signup_app, prefix='signup_app'),
                   pd.get_dummies(train.first_device_type, prefix='first_device_type'),
               pd.get_dummies(train.first_browser, prefix='first_browser'),
                pd.get_dummies(train.date_first_booking_m, prefix='date_first_booking_m'),
                  pd.get_dummies(train.date_first_booking_y , prefix='date_first_booking_y '),
                   pd.get_dummies(train.date_account_created_m, prefix='date_account_created_m'),
                   pd.get_dummies(train.date_account_created_y , prefix='date_account_created_y'),
                   pd.get_dummies(train.age1 , prefix='age1'),train.signup_flow
                   ], axis=1)
train_x.head()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(train.country_destination)
le.classes_
train_y=le.transform(train.country_destination) 
train_x
###还需要将字符型的转为dummy
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn import tree
clf = tree.DecisionTreeClassifier()
gnb = MultinomialNB()
clf = linear_model.LogisticRegression(C=1e5)
clf.fit(train_x, train_y)
y_pred = clf.predict(train_x)
print("Number of mislabeled points out of a total %d points : %d"  % (train_x.shape[0],(train_y != y_pred).sum()))
y_pred = clf.predict_proba(train_x)  

#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
test=pd.read_csv('../input/test_users.csv')
id_test = test['id']
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('sub.csv',index=False)
