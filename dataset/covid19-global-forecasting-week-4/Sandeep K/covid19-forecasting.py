# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# Predict COVID19 Fatalities, Confirmed cases
# Import python packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

# import train data
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

# import test data
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')

# Query sample train data
train.head()

train.info()
# Id is int. Province_State, Country_Region and Date are categorical. 
# ConfirmedCases and Fatalities are float

# Describe data to see count, mean and std...
train.describe()

#Check the percentage of null values per variable
train.isnull().sum()/train.shape[0]*100
test.isnull().sum()/test.shape[0]*100
#Province_State has 57% null values

# impute Province_State null values with common value 'Province'

train['Province_State'].fillna('Province', inplace = True)
test['Province_State'].fillna('Province', inplace = True)

#Seaborn formatting
sns.set()

# REMOVE SEABORN FORMATTING
#sns.reset_orig()
#EDA
#
#Distribution of target variables
#plt.style.use('default')
#plt.figure(figsize=(8,6))
#sns.distplot(train.ConfirmedCases, bins=15)
#plt.ticklabel_format(style='plain', axis='x', scilimits=(0,1))
#Scatter plot between 
#plt.scatter(train['Date'],train['Fatalities'], c='red', s=1)

# Encode categorical variables
le = LabelEncoder()
dep_var = ['Province_State','Country_Region','Date']

for col in dep_var:
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])
    
# Drop Id column from train
train.drop(['Id'], axis=1, inplace=True)
test.drop(['ForecastId'], axis=1, inplace=True)
    
train.hist(bins=15, figsize=(10,15))

corr = train.corr()

corr['Fatalities'].sort_values(ascending = False)
corr['ConfirmedCases'].sort_values(ascending = False)

# independent variable
X_train=train[['Province_State','Country_Region','Date']]
# Dependent variable
y_fatal_train=train[['Fatalities']]
y_conf_train=train[['ConfirmedCases']]

# independent variable
X_test=test[['Province_State','Country_Region','Date']]


# Predict fatalities
#X_train_fatal, X_test_fatal, y_train_fatal, y_test_fatal = train_test_split(X, y_fatal, test_size = 0.2, random_state = 0)

k = 3
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_fatal_train)
neigh


y_fatal_pred = neigh.predict(X_test)

# Predict confirmed cases
k = 3
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_conf_train)
neigh


y_conf_pred = neigh.predict(X_test)

# Submission
submission['ConfirmedCases']=y_conf_pred
submission['Fatalities']=y_fatal_pred

submission.to_csv('submission.csv', index=False)