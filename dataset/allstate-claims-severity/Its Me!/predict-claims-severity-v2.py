# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Read data
train = pd.read_csv('../input/train.csv', nrows=2000)
test = pd.read_csv('../input/test.csv', nrows=2000)


#Category variables 116
train_cat = train.ix[:,'cat1':'cat116']
test_cat = train.ix[:,'cat1':'cat116']
y_train = train['loss']

#Continuous variable 14
train_cont = test.ix[:,'cont1':'cont14']
test_cont = test.ix[:,'cont1':'cont14']

#Convert categorical features using  dummy variables
train_cat_dummy = pd.get_dummies(train_cat)
print(train_cat_dummy.head(5))
test_cat_dummy = pd.get_dummies(test_cat)
print(test_cat_dummy.head(5))

# Join categorical and continuous data
X_train = train_cat_dummy.join(train_cont)
X_test = test_cat_dummy.join(test_cont)

#Using decisionTreeRegressor##
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=100, random_state=0)
tree.fit(X_train,y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
y_predtree = tree.predict(X_test)
print("DecisionTreeRegressor")
print(y_predtree[0])

#RandomForest regressor
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=50, random_state=3)
rfr.fit(X_train,y_train)
print("Accuracy on training set: {:.3f}".format(rfr.score(X_train, y_train)))
y_predrfr = rfr.predict(X_test)
print("RandomForestRegressor")
print(y_predrfr[0])

#Pick the better prediction
if (y_predtree[0]>y_predrfr[0]):
    ypred=y_predrfr
    print("Selecting --> RandomForestRegressor")
else:
    ypred=y_predtree
    print("Selecting --> DecisionTreeRegressor")

preds = pd.DataFrame({"id": test['id'],"loss": ypred})
preds.head(5)
preds.to_csv('AllStateClaimsSeverity_yyyymmdd.csv', index=False)








