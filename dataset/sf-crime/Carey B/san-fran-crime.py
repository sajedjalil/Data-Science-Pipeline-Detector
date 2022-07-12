
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Import Libraries
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Get data
train = pd.read_csv('../input/train.csv', parse_dates = ['Dates'])
test = pd.read_csv('../input/test.csv', parse_dates = ['Dates'])

# Need to clean up outliers
#Since neither X or Y are not Guassian we will use interquantile range
# For X
IQR = train.X.quantile(0.75) - train.X.quantile(0.25)
Lower_fence_X = train.X.quantile(0.25) - (IQR * 3)
Upper_fence_X = train.X.quantile(0.75) + (IQR * 3)
train.loc[train.X < -122.51093037786198, 'X']= -122.51093037786198
train.loc[train.X > -122.32897987265702, 'X']= -122.32897987265702

# For Y
IQR = train.Y.quantile(0.75) - train.Y.quantile(0.25)
Lower_fence_Y = train.Y.quantile(0.25) - (IQR * 3)
Upper_fence_Y = train.Y.quantile(0.75) + (IQR * 3)
train.loc[train.Y > 37.8801919977151, 'Y']= 37.8801919977151


# Encode the train features and set up the train_data
# Convert Category crime labels to numbers. This will be our model targets
category = LabelEncoder()
crime = category.fit_transform(train.Category)
# One Hot encode districts (PdDistrict)
districts = pd.get_dummies(train.PdDistrict)
# One Hot encode DayOfWeek
day_of_week = pd.get_dummies(train.DayOfWeek)
# One Hot encode hours
hour = train.Dates.dt.hour
hours = pd.get_dummies(hour, prefix='Hour')
# One Hot encode month
month = train.Dates.dt.month
months = pd.get_dummies(month, prefix='Month')
# Create a training data set with the above one hot encodings
train_data = pd.concat([hours, day_of_week, months, districts], axis=1)
# For Address, there are two types, those with 'Block' and those with a '/'.
# We add one new column to contain either 0 or 1 (and  avoid Duplicate Dummy trap)
train_data['Block'] = train['Address'].str.contains('Block').map(int)
# Now encode Address with Category counts
# Build a dictionary for Address
address_dict = train['Address'].value_counts().to_dict()
# Add Address to Train
train_data.loc[:,'Address'] = train.loc[:, 'Address'].map(address_dict)
# Add Lat and Log to the train data
train_data["X"] = train['X']
train_data["Y"] = train['Y']
# Identify the evening hour, with the expectation that evening hours have more crime
train_data['Evening']=hour
train_data['Evening']=train_data['Evening'].apply(lambda x: train_data['Evening'][x] in range (18,23))
train_data['Evening']=train_data['Evening'].apply(lambda x: int(bool(x)))
# Need to add crime to train_data
train_data['Crime'] = crime


# Encode the test features and set up the test_data
# Since neither X or Y are not Guassian we will use interquantile range
# For X
IQR = test.X.quantile(0.75) - test.X.quantile(0.25)
Lower_fence_X = test.X.quantile(0.25) - (IQR * 3)
Upper_fence_X = test.X.quantile(0.75) + (IQR * 3)
# Since we do not have many we will do top-encoding and bottom-encoding
test.loc[test.X < -122.51093037786198, 'X']= -122.51093037786198
test.loc[test.X > -122.32897987265702, 'X']= -122.32897987265702
# For Y
IQR = test.Y.quantile(0.75) - test.Y.quantile(0.25)
Lower_fence_Y = test.Y.quantile(0.25) - (IQR * 3)
Upper_fence_Y = test.Y.quantile(0.75) + (IQR * 3)
# Since we do not have many we will do top-encoding
test.loc[test.Y > 37.8801919977151, 'Y']= 37.8801919977151
# One Hot encode districts (PdDistrict)
districts = pd.get_dummies(test.PdDistrict)
# One Hot encode DayOfWeek
day_of_week = pd.get_dummies(test.DayOfWeek)
# One Hot encode hours
hour = test.Dates.dt.hour
hours = pd.get_dummies(hour, prefix='Hour') 
# One Hot encode month
month = test.Dates.dt.month
months = pd.get_dummies(month, prefix='Month')
# Create a testing data set with the above one hot encodings
test_data = pd.concat([hours, day_of_week, months, districts], axis=1)
# For Address, there are two types, those with 'Block' and those with a '/'.
# We add one new column to contain either 0 or 1 (and  avoid Duplicate Dummy trap)
test_data['Block'] = test['Address'].str.contains('Block').map(int)
# Now encode Address with Category counts
# Build a dictionary for Address
address_dict = test['Address'].value_counts().to_dict()
# Add Address to test
test_data.loc[:,'Address'] = test.loc[:, 'Address'].map(address_dict)
# Add Lat and Long to the test_data
test_data["X"] = test['X']
test_data["Y"] = test['Y']
# Identify the evening hour, with the expectation that evening hours have more crime
test_data['Evening']=hour
test_data['Evening']=test_data['Evening'].apply(lambda x: test_data['Evening'][x] in range (18,23))
test_data['Evening']=test_data['Evening'].apply(lambda x: int(bool(x)))

# Split train_data for testing
training, validation = train_test_split(train_data, train_size=.60, random_state=10)

# Set the features for non-tree models. 
# Remove one hot encoded first columns (Hour_0, Friday, Month_1, Year_1, and BAYVIEW) 
# to Avoid Dummy Variable Trap
features = ['Hour_1', 'Hour_2', 'Hour_3', 'Hour_4', 'Hour_5', 'Hour_6',
            'Hour_7', 'Hour_8', 'Hour_9', 'Hour_10', 'Hour_11', 'Hour_12',
            'Hour_13', 'Hour_14', 'Hour_15', 'Hour_16', 'Hour_17', 'Hour_18',
            'Hour_19', 'Hour_20', 'Hour_21', 'Hour_22', 'Hour_23', 
            'Evening',
            'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday',
            'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6',
            'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12',
            'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND',
            'SOUTHERN', 'TARAVAL', 'TENDERLOIN', 'Block', 'Address', 'X', 'Y']

# Full feature list for tree models - Note this submission is use this for Random Forest
features_tree = ['Hour_0', 'Hour_1', 'Hour_2', 'Hour_3', 'Hour_4', 'Hour_5', 'Hour_6',
                 'Hour_7', 'Hour_8', 'Hour_9', 'Hour_10', 'Hour_11', 'Hour_12',
                 'Hour_13', 'Hour_14', 'Hour_15', 'Hour_16', 'Hour_17', 'Hour_18',
                 'Hour_19', 'Hour_20', 'Hour_21', 'Hour_22', 'Hour_23', 
                 'Evening',
                 'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday',
                 'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6',
                 'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12',
                 'BAYVIEW',
                 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND',
                 'SOUTHERN', 'TARAVAL', 'TENDERLOIN', 'Block', 'Address', 'X', 'Y']

# fit a standart scaler to the training set. We will transform it for  non-tree models
scaler = StandardScaler()
scaler.fit(training[features]) 

### GridSearch to find Random Forest parms
# Note, this takes a very long time to complete
# rfc_model = RandomForestClassifier(n_jobs=10, random_state=100) 
# param_grid = {"n_estimators" : [100, 200, 300],
#               "max_depth" : [10, 13, 16, 20],
#               "min_samples_leaf" : [1, 2, 4]}
 
# rfc_grid = GridSearchCV(estimator=rfc_model, param_grid=param_grid, cv= 10)
# rfc_grid.fit(training[features], training['Crime'])
# print(rfc_grid.best_params_)
###

# Random Forest model used for submission
# Note, n_jobs = 10
rf_model = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf = 1, n_jobs=10, random_state=10)
rf_model.fit(training[features_tree], training['Crime'])

predicted = rf_model.predict_proba(validation[features_tree])
predicted_loss = np.array(predicted)
print(log_loss(validation['Crime'], predicted_loss))
# Log_Loss = 2.3616388429056436

# Now for the submission
# Predict the test data for Submission
predicted_sub = rf_model.predict_proba(test_data[features_tree])
# Create the submission file
submission_results = pd.DataFrame(predicted_sub, columns=category.classes_)
submission_results.to_csv('submission.csv', index_label = 'Id' )

# BernoulliNB
# bnb_model = BernoulliNB()
# bnb_model.fit(scaler.transform(training[features]), training['Crime'])
# predicted = np.array(bnb_model.predict_proba(scaler.transform(validation[features])))
# print(log_loss(validation['Crime'], predicted))

#Logistic Regression
# logit_model = LogisticRegression(C=.01)
# logit_model.fit(scaler.transform(training[features]), training['Crime'])
# predicted = np.array(logit_model.predict_proba(scaler.transform(validation[features])))
# print(log_loss(validation['Crime'], predicted))



# Any results you write to the current directory are saved as output.