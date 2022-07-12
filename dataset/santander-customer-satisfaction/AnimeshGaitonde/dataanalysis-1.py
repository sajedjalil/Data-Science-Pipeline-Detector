# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import cross_validation #sklearn for machine learning 
# Import the logistic regression class
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df1 = pd.read_csv('../input/train.csv')
print (df1.columns)
print (type(df1.columns))
predictors = []
for i in df1.columns:
    predictors.append(i)
predictors.remove('TARGET')
#print (predictors)
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
#scores = cross_validation.cross_val_score(alg, df1[predictors], df1['TARGET'], cv=3)
# Take the mean of the scores (because we have one for each fold)
#print(scores.mean())
# Any results you write to the current directory are saved as output.
df2 = pd.read_csv('../input/test.csv')
print (df2.columns)
#print (df2.isnull().sum())
#alg.fit(df1[predictors], df1['TARGET'])
#predictions = alg.predict(df2[predictors])

clf = linear_model.SGDClassifier()
clf.fit(df1[predictors],df1['TARGET'])

predictions = clf.predict(df2[predictors])
submission = pd.DataFrame({"ID":df2["ID"],"TARGET":predictions})
submission = submission[['ID','TARGET']]
submission.to_csv('submission_sgd_classifies.csv')
print (submission.head(10))
sample_submissions = pd.read_csv('../input/sample_submission.csv')
sample_submissions.tail(20)
