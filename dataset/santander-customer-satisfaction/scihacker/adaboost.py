# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

x_train = pd.read_csv("../input/train.csv")
x_test = pd.read_csv("../input/test.csv").values

target = x_train.iloc[:, -1].values
X = x_train.iloc[:, 1:-1].values

# This 45 is calculated by cross_validation, on the AUC value.
dt = DecisionTreeClassifier(class_weight={1:45})
clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)
clf.fit(X, target)
result = clf.predict_proba(x_test[:, 1:])[:, 1]

result = pd.DataFrame({"ID": x_test[:, 0].astype('int'), "TARGET": result})
result.to_csv('submission.csv', index=False)