# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 19:47:06 2019

@author: Arrow
"""
import numpy as np
import pandas as pd
from sklearn import ensemble
#from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the training and test data sets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train['ele_vert'] = train.Elevation-train.Vertical_Distance_To_Hydrology

train['slope_hyd'] = (train['Horizontal_Distance_To_Hydrology']**2+
     train['Vertical_Distance_To_Hydrology']**2)**0.5
train.slope_hyd=train.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any
print(train.head)

test['ele_vert'] = test.Elevation-test.Vertical_Distance_To_Hydrology

test['slope_hyd'] = (test['Horizontal_Distance_To_Hydrology']**2+
    test['Vertical_Distance_To_Hydrology']**2)**0.5
test.slope_hyd=test.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any


# Create numpy arrays for use with scikit-learn
train_X = train.drop(['Id','Cover_Type','Horizontal_Distance_To_Hydrology',
                'Vertical_Distance_To_Hydrology','Elevation'],axis=1).values
train_y = train.Cover_Type.values
test_X = test.drop(['Id','Horizontal_Distance_To_Hydrology',
                'Vertical_Distance_To_Hydrology','Elevation'],axis=1).values

# Split the training set into training and validation sets
X,X_,y,y_ = train_test_split(train_X,train_y,test_size=0.2)

# Train and predict with the random forest classifier
rf = ensemble.RandomForestClassifier()
rf.fit(X,y)
y_rf = rf.predict(X_)
print (metrics.classification_report(y_,y_rf))
print (metrics.accuracy_score(y_,y_rf))

# Retrain with entire training set and predict test set.
rf.fit(train_X,train_y)
y_test_rf = rf.predict(test_X)

# Write to CSV
pd.DataFrame({'Id':test.Id.values,'Cover_Type':y_test_rf})\
            .sort_index(ascending=False,axis=1).to_csv('rf1.csv',index=False)