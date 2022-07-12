#Disclaimer: newbie's code (2.66 in LB)
######################################################################################################
#                                       Import Libraries and Data
######################################################################################################

print("Importing libraries and data")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier


from datetime import datetime

df_train = pd.read_csv("../input/train.csv",index_col=None)
df_test = pd.read_csv("../input/test.csv",index_col=None)

crime_cat = df_train.Category.value_counts().index


######################################################################################################
#                                       Make Feature Columns
######################################################################################################

print("Features")

print("Making X strictly positive")
df_train["X"] = df_train.X.map(lambda x: -x)
df_test["X"] = df_test.X.map(lambda x: -x)

def is_weekend(day):
    if day in ["Friday","Saturday","Sunday"]:
        return(1)
    else:
        return(0)
    

def get_hour_norm(d):
    _ = datetime.strptime(d,'%Y-%m-%d %H:%M:%S').hour
    return(_/24.0)

def get_month_norm(d):
    _ = datetime.strptime(d,'%Y-%m-%d %H:%M:%S').month
    return(_/12.0)


print("Hour feature")
df_train.loc[:,"Hour"] = df_train.Dates.map(get_hour_norm)
df_test.loc[:,"Hour"] = df_test.Dates.map(get_hour_norm)

print("Month feature")
df_train.loc[:,"Month"] = df_train.Dates.map(get_month_norm)
df_test.loc[:,"Month"] = df_test.Dates.map(get_month_norm)

print("Year feature")
df_train.loc[:,"Year"] = df_train.Dates.map(lambda d: datetime.strptime(d,'%Y-%m-%d %H:%M:%S').year)
df_test.loc[:,"Year"] = df_test.Dates.map(lambda d: datetime.strptime(d,'%Y-%m-%d %H:%M:%S').year)

#df_train.loc[:,"IsWeekend"] = df_train.DayOfWeek.map(is_weekend)
#df_test.loc[:,"IsWeekend"] = df_test.DayOfWeek.map(is_weekend)

print("Keeping used features")
print("Categories (y variable)")
df_train_pred = df_train.loc[:,["Year","Category"]]

df_train = df_train.loc[:,["Year","Hour","Month","DayOfWeek","X","Y"]]
print("Creating dummies for train data")
df_train = pd.get_dummies(df_train)

df_test = df_test.loc[:,["Id","Year","Hour","Month","DayOfWeek","X","Y"]]
print("Creating dummies for test data")
df_test = pd.get_dummies(df_test)

######################################################################################################
#                                           Train Models
######################################################################################################


X = df_train.ix[:,1:].values.tolist()
y = df_train_pred.ix[:,"Category"].values.tolist()

print("Training kernel approximation")

rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X)

print("Fitting SV classifier")
clf = SGDClassifier(loss='log', penalty = 'elasticnet', alpha = 0.001,l1_ratio = 1)

clf.fit(X_features, y)

print("Predicting categories")

X = df_test.ix[:,2:].values.tolist()
X_features = rbf_feature.transform(X)

print("Exporting results")

res = pd.DataFrame(clf.predict_proba(X_features),columns = clf.classes_, index = df_test.Id)
res.index.name = "Id"

print(res)

res = res.round(decimals=5)

res.to_csv('time_only_RBF_SV_proba.csv')

