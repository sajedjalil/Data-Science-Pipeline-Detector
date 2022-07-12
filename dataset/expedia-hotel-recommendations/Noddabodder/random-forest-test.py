
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import mixture
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
import pandas as pd

destinations = pd.read_csv("../input/destinations.csv", nrows=1000000, error_bad_lines=False)
test = pd.read_csv("../input/test.csv", nrows=1000000, error_bad_lines=False)
train = pd.read_csv("../input/train.csv", nrows=1000000, 
                   error_bad_lines=False)

from datetime import datetime
del train["date_time"]
del train["srch_ci"]
del train["srch_co"]
del train["is_booking"]
del train["cnt"]

train = train.fillna(0)
test = test.fillna(0)

predictors = [c for c in train.columns if c not in ["hotel_cluster"]]
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
#scores = cross_validation.cross_val_score(clf, train[predictors], train['hotel_cluster'], cv=3)
clf.fit(train[predictors], train['hotel_cluster'])
testing = clf.predict(test[predictors])
#test = clf.predict_proba( test[predictors], test['hotel_cluster'])
print (len(testing))
"""

#datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')

most_common_clusters = list(train.hotel_cluster.value_counts().head().index)
predictions = [most_common_clusters for i in range(train.shape[0])]

import ml_metrics as metrics
target = [[l] for l in train["hotel_cluster"]]
print (metrics.mapk(target, predictions, k=5))

predictors = [c for c in train.columns if c not in ["hotel_cluster"]]
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, min_weight_fraction_leaf=0.1)
scores = cross_validation.cross_val_score(clf, train[predictors], train['hotel_cluster'], cv=5)
print (scores)"""