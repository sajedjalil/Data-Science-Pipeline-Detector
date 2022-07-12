# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../input/train.csv')


dataset['v18q1']=dataset['v18q1'].fillna(0)
dataset['rez_esc']=dataset['rez_esc'].fillna(0)
dataset=dataset.drop(['dependency','edjefe','edjefa','idhogar'],1)


x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 138].values

dataset_x=pd.DataFrame(x)

#is there any missing values ??


#print(dataset.isna().any())
report1=dataset_x.isnull().sum()
null_columns=dataset_x.columns[dataset_x.isnull().any()]
null_val=dataset_x[null_columns].isnull().sum()

#handle the NaN values!!
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 0:1])
x[:, 0:1] = imputer.transform(x[:, 0:1])

imputer1 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer1 = imputer1.fit(x[:, 135:136])
x[:, 135:136] = imputer1.transform(x[:, 135:136])

imputer = imputer.fit(x[:, 98:99])
x[:, 98:99] = imputer.transform(x[:, 98:99])

 #splitting the dataset into train and test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Fitting XGBoost to the Training set

import xgboost
classifier = xgboost.XGBClassifier()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

dataset_ypred=pd.DataFrame(y_pred)




# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#the test dataset
dataset2 = pd.read_csv('../input/test.csv')
dataset2['v18q1']=dataset2['v18q1'].fillna(0)
dataset2['rez_esc']=dataset2['rez_esc'].fillna(0)
dataset2=dataset2.drop(['dependency','edjefe','edjefa','idhogar'],1)
dataset_ids=dataset2["Id"]

testing = dataset2.iloc[:, 1:138].values
testing_df=pd.DataFrame(testing)


report2=testing_df.isnull().sum()
null_columns2=testing_df.columns[testing_df.isnull().any()]
null_val2=testing_df[null_columns2].isnull().sum()

imputer = imputer.fit(testing[:, 0:1])
testing[:, 0:1] = imputer.transform(testing[:, 0:1])

imputer = imputer.fit(testing[:, 98:99])
testing[:, 98:99] = imputer.transform(testing[:, 98:99])

imputer1 = imputer1.fit(testing[:, 135:136])
testing[:, 135:136] = imputer1.transform(testing[:, 135:136])

test_pred = classifier.predict(testing)
sample=pd.DataFrame(test_pred)
frame=[dataset_ids,sample]
result=pd.concat(frame,axis=1)
#result.to_csv("sample_submission.csv",index=False)



classifier1 = xgboost.XGBClassifier()
classifier1.fit(x,y)

yy_pred=classifier1.predict(testing)
sample_testing=pd.DataFrame(yy_pred)

frame2=[dataset_ids,sample_testing]
result2=pd.concat(frame2,axis=1)
#result2.to_csv("sample_submission.csv",index=False)

sbmt = pd.DataFrame({'Id':dataset_ids, 'Target': yy_pred})
sbmt.to_csv('sample_submission.csv', index=False)















