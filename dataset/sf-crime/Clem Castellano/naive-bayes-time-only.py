
######################################################################################################
#                                       Import Libraries and Data
######################################################################################################

print("Importing libraries and data")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.naive_bayes import MultinomialNB


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

df_train = df_train.loc[:,["Year","Hour","Month","DayOfWeek"]]
print("Creating dummies for train data")
df_train = pd.get_dummies(df_train)

df_test = df_test.loc[:,["Id","Year","Hour","Month","DayOfWeek"]]
print("Creating dummies for test data")
df_test = pd.get_dummies(df_test)

######################################################################################################
#                                           Train Models
######################################################################################################

# Reminder: self imposed constraint = avoid look-ahead bias
# I do not use future years data for predictions

number_of_crimes = df_train_pred.Category.value_counts()
avg_crimes = number_of_crimes/sum(number_of_crimes)

clf = MultinomialNB(class_prior = avg_crimes.values.tolist())

X = df_train.ix[df_train.Year == 2003,1:].values.tolist()
y = df_train_pred.ix[df_train_pred.Year == 2003,"Category"].values.tolist()

clf.partial_fit(X, y, classes=crime_cat)


probs = []
for y in [2003,2004]:
    X = df_test.ix[df_test.Year == y,2:].values.tolist()
    tmp = clf.predict_proba(X)
    tmp = pd.DataFrame(tmp, columns = crime_cat)
    tmp.index = df_test.ix[df_test.Year == y,"Id"]
    probs.append(tmp)

year_pairs = []
for i in range(11):
    year_pairs.append([2004+i,2005+i])

for y in year_pairs:
    y1 = y[0]
    y2 = y[1]
    print("Years: {} (train) and {} (test)".format(y1,y2))
    print("Training model")
    X = df_train.ix[df_train.Year == y1,1:].values.tolist()
    Y = df_train_pred.ix[df_train_pred.Year == y1,"Category"].values.tolist()
    clf.partial_fit(X, Y)
    X = df_test.ix[df_test.Year == y2,2:].values.tolist()
    tmp = clf.predict_proba(X)
    tmp = pd.DataFrame(tmp, columns = crime_cat)
    tmp.index = df_test.ix[df_test.Year == y2,"Id"]
    probs.append(tmp)


probs = pd.concat(probs)
probs = probs.fillna(0)
probs = probs.round(decimals=2)
probs.index.name = "Id"


print(probs)

probs.to_csv('time_only_NB_proba.csv')



