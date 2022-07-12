import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from xgboost.sklearn import XGBClassifier
import xgboost as xgb

def predict(model, ClassificationAlg,testX,testY):
    print("\n\n"+ClassificationAlg)
    PredictedY = model.predict(testX)
    print(classification_report(testY, PredictedY))
    print('Confusion Matrix\n')
    print(confusion_matrix(testY, PredictedY))
    print("\nThe accuracy score is {:.2%}".format(accuracy_score(testY, PredictedY)))


def Random_Forest(trainX,testX,trainY,testY):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(max_depth=5)
    rf.fit(trainX, trainY)
    predict(rf,"Random Forest",testX,testY)

def AdaBoost(trainX,testX,trainY,testY):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    adaBoost = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=100,
        learning_rate=0.1,
        random_state=7
    )
    adaBoost.fit(trainX, trainY)
    predict(adaBoost, "AdaBoost",testX,testY)


def Naive_Bayes(trainX,testX,trainY,testY):
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()
    nb.fit(trainX, trainY)
    predict(nb, "Naive Bayes", testX, testY)


def XGBoost(trainX,testX,trainY,testY):
    from xgboost.sklearn import XGBClassifier
    # fit model no training data
    model = XGBClassifier()
    model.fit(trainX, trainY)
    predict(model, "XGBoost", testX, testY)

# Get the data from files
data = pd.read_csv("../input/train_users_2.csv")
session_data = pd.read_csv("../input/sessions.csv")

# Replace the missing value with NaN
session_data.device_type.replace('-unknown-',np.nan, inplace = True)

# Extract the aggregated value from Session records
# total activity
activity_count = session_data.groupby('user_id').agg({'secs_elapsed':'count'}).reset_index()
activity_count=activity_count.rename(columns = {'user_id':'id', 'secs_elapsed':'total_activity'})
# Merge with source
data = data.merge(activity_count, how='left', on=('id'))
data.total_activity.replace(np.nan, 0, inplace = True)
# total time spent
activity_time = session_data.groupby('user_id').agg({'secs_elapsed':'mean'}).reset_index()
activity_time=activity_time.rename(columns = {'user_id':'id', 'secs_elapsed':'total_time'})
# Merge with source
data = data.merge(activity_time, how='left', on=('id'))
data.total_time.replace(np.nan, 0, inplace = True)

session_data_backup = session_data.copy(deep=True)
session_data.drop('action',axis=1, inplace = True)
session_data.drop('action_type',axis=1, inplace = True)
session_data.drop('action_detail',axis=1, inplace = True)
session_data.drop('secs_elapsed',axis=1, inplace = True)
session_data['value'] = 1

# Pivot the source to extract the device type
devices_used = session_data.pivot_table(index='user_id', columns='device_type', values='value', aggfunc='mean').reset_index()
devices_used=devices_used.rename(columns = {'user_id':'id'})
# Merge with the source
data = data.merge(devices_used, how='left', on=('id'))

# Update the age
data.gender.replace("-unknown-", np.nan, inplace = True)
data.drop('id', axis=1, inplace = True)
data.loc[data.age > 90, 'age'] = np.nan
data.loc[data.age < 14, 'age'] = np.nan
data.age = data.age%10

data.drop('date_account_created', axis=1, inplace = True)
data.drop('date_first_booking', axis=1, inplace = True)
data.drop('timestamp_first_active', axis=1, inplace = True)

# encode the data
le = preprocessing.LabelEncoder()
for categorical_feature in data.columns.values:
    data[categorical_feature] = data[categorical_feature].factorize()[0]
    data[categorical_feature] = le.fit_transform(data[categorical_feature])

y_train = data['country_destination']
x_train = data.drop('country_destination', axis=1)

trainX,testX,trainY,testY = train_test_split(x_train,y_train,test_size=.20)
AdaBoost(trainX,testX,trainY,testY)
Naive_Bayes(trainX,testX,trainY,testY)
Random_Forest(trainX,testX,trainY,testY)
XGBoost(trainX,testX,trainY,testY)


