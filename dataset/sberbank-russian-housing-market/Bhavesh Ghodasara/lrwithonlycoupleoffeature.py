# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn import model_selection, preprocessing

# Reading all three files
train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])

# Merging macro data with train and test
train = pd.merge(train, macro, how='left', on='timestamp')
test = pd.merge(test, macro, how='left', on='timestamp')

# normalize prize feature
train["price_doc"] = np.log1p(train["price_doc"])
# store it as Y
Y_train = train["price_doc"]

# Dropping price column
train.drop("price_doc", axis=1, inplace=True)

train_limited = train[['id','sub_area','product_type','ecology','full_sq']].copy()
test_limited = test[['id','sub_area','product_type','ecology','full_sq']].copy()


X_train = train_limited
X_test  = test_limited

# Create dummy variables
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

X_train.drop("sub_area_Poselenie Klenovskoe", axis=1, inplace=True)

print(X_train.shape)
print(X_test.shape)

# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
result= regr.fit(X_train, Y_train)
Pred_train = regr.predict(X_train)
Y_pred = np.expm1(regr.predict(X_test))


# Preparing submission file
submission = pd.DataFrame({"id": test["id"],"price_doc": Y_pred})
submission.to_csv('submission.csv', index=False)