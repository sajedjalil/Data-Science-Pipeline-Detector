# In this kernel, I run the model with time features only derived from this kernel: 
# https://www.kaggle.com/lesibius/crime-scene-exploration-and-knn-fit

#I'll use a KNN to fit the data

######################################################################################################
#                                       Import Libraries and Data
######################################################################################################

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import KNeighborsClassifier

from datetime import datetime

df_train = pd.read_csv("../input/train.csv",index_col=None)
df_test = pd.read_csv("../input/test.csv",index_col=None)


# Parameter of the model: number of neighbor
# I derived this figure from a cross validation not shown in Kaggle (too long to run)
n_neighbors = 30

######################################################################################################
#                                       Make Feature Columns
######################################################################################################

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



df_train.loc[:,"Hour"] = df_train.Dates.map(get_hour_norm)
df_test.loc[:,"Hour"] = df_test.Dates.map(get_hour_norm)

df_train.loc[:,"Month"] = df_train.Dates.map(get_month_norm)
df_test.loc[:,"Month"] = df_test.Dates.map(get_month_norm)

df_train.loc[:,"Year"] = df_train.Dates.map(lambda d: datetime.strptime(d,'%Y-%m-%d %H:%M:%S').year)
df_test.loc[:,"Year"] = df_test.Dates.map(lambda d: datetime.strptime(d,'%Y-%m-%d %H:%M:%S').year)

df_train.loc[:,"IsWeekend"] = df_train.DayOfWeek.map(is_weekend)
df_test.loc[:,"IsWeekend"] = df_test.DayOfWeek.map(is_weekend)

df_test["Prediction"] = 'AAA'

######################################################################################################
#                                           Train Models
######################################################################################################

year_pairs = [[2003,2003]]
for i in range(12):
    year_pairs.append([2003+i,2004+i])


years = df_train.Year.value_counts().index

for y in year_pairs:
    y1 = y[0]
    y2 = y[1]
    print("Years: {} (train) and {} (test)".format(y1,y2))
    print("Training model")
    knn = KNeighborsClassifier(n_neighbors = 30)
    print("Predicting")
    knn.fit(df_train[df_train["Year"] == y1].loc[:,["Hour","IsWeekend","Month"]],df_train[df_train["Year"] == y1].loc[:,"Category"])
    res =  knn.predict(df_test[df_test["Year"] == y2].loc[:,["Hour","IsWeekend","Month"]])
    print(res)
    df_test.ix[df_test.Year == y2,"Prediction"] = res
    
df_res = pd.get_dummies(df_test.Prediction)
df_res.index = df_test.Id

crime_cat = df_train.Category.value_counts().index

for c in crime_cat:
    if not c in df_res.columns.values:
        df_res[c] = 0


df_res.to_csv('time_only_submission.csv')







