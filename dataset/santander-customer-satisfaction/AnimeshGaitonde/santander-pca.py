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
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC


# Any results you write to the current directory are saved as output.
df1 = pd.read_csv('../input/train.csv')
#print (df1.columns)
#print (type(df1.columns))
predictors = []
for i in df1.columns:
    predictors.append(i)
predictors.remove('TARGET')
#print (predictors)

df2 = pd.read_csv('../input/test.csv')
print (df2.columns)

# Extract the top 50 principal components in features
n_components = 50
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(df1[predictors])

# Transofrm the data using PCA
X_train_pca = pca.transform(df1[predictors])
X_test_pca = pca.transform(df2[predictors])


clf = LogisticRegression()
clf.fit(X_train_pca,df1['TARGET'])

predictions = clf.predict(X_test_pca)
submission = pd.DataFrame({"ID":df2["ID"],"TARGET":predictions})
submission = submission[['ID','TARGET']]
submission.to_csv('submission_PCA_Logistic.csv')
