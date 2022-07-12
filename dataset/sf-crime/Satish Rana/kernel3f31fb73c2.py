# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 12:34:13 2018

@author: rana220119
@description : Model building using Random Forest
"""

import pandas as pd

dataset = pd.read_csv('../input/train.csv')

#To check null - there is no missing data  
dataset.isnull().sum()

dataset.drop_duplicates(inplace = True)

# Handling for outliers for lattitude and langitude
dataset['Y']=dataset['Y'].apply(lambda x : x if 37.82 > x else 37.82 )
dataset['X']=dataset['X'].apply(lambda x : x if -122.3 > x else -122.3 )

dataset['dateobj'] = pd.to_datetime(dataset['Dates'])
dataset['year'] = dataset.dateobj.dt.year
dataset['month'] = dataset.dateobj.dt.month
dataset['hour'] = dataset.dateobj.dt.hour

#dropping unwanted columns
drop_columns = ['Dates','dateobj','Descript','Resolution','Address']
dataset.drop(drop_columns,axis=1,inplace=True)

# removing the category which are having lesser signifiance count
col_values = ['LARCENY/THEFT','OTHER OFFENSES','NON-CRIMINAL','ASSAULT','DRUG/NARCOTIC','VEHICLE THEFT','VANDALISM','WARRANTS','BURGLARY']
dataset = dataset[dataset['Category'].isin(col_values)]


X = dataset.drop(['Category'],axis=1) 
Y = dataset['Category'] 

X = pd.get_dummies(X, columns = ['DayOfWeek','PdDistrict','year','month','hour'], drop_first=True)

# Split into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=1)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 20)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#accurancy of model
from sklearn.metrics import accuracy_score
print("model accruacy ",accuracy_score(Y_test,y_pred)*100)
