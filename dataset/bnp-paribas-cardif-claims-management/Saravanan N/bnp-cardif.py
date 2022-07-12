# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import ShuffleSplit, train_test_split
from sklearn.metrics import mean_squared_error

data = data.select_dtypes(exclude=[object])
data.fillna(-999, inplace=True)
features = data.ix[:, "v1":"v131"]
labels = data.ix[:, "target"]
selected_features = [
    "v12",
    "v14",
    "v21",
    "v34",
    "v50",
]
#features = data.ix[:, selected_features]
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, 
                                                                            test_size=0.3)
#clf = DecisionTreeClassifier()
#clf.fit(features_train, labels_train)
#a = clf.feature_importances_
#for index, i in enumerate(a):
#    if i > 0.05:
#        print(features_train.columns[index])
clf = AdaBoostClassifier()
#clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict_proba(features_test)
#print(clf.classes_)

# import xgboost as xgb
# data_train = xgb.DMatrix(features_train.as_matrix(), label=labels_train.as_matrix())
# params ={'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
# clf = xgb.train(params, data_train)
# features_test = xgb.DMatrix(features_test)
# pred = clf.predict(features_test)
print(mean_squared_error(labels_test, pred[:,1])**0.5)

test_data = pd.read_csv("../input/test.csv")
test_data = test_data.select_dtypes(exclude=[object])
test_data.fillna(-999, inplace=True)
features = test_data.ix[:, "v1":"v131"]
ids = test_data.ix[:, "ID"]
#test_features = xgb.DMatrix(features)
test_features = features.as_matrix()
pred = clf.predict_proba(test_features)
output_data = pd.concat([ids, pd.DataFrame(pred[:,1], columns=["PredictedProb"])], axis=1)
output_data.to_csv("output.csv", index=False)



