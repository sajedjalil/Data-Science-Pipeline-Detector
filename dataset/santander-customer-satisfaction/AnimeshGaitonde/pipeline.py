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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn import cross_validation, grid_search
from sklearn.pipeline import Pipeline
from sklearn import linear_model, decomposition
df1 = pd.read_csv('../input/train.csv')
#print (df1.columns)
#print (type(df1.columns))
predictors = []
for i in df1.columns:
    predictors.append(i)
predictors.remove('TARGET')
df2 = pd.read_csv('../input/test.csv')
#print (df2.columns)
X_train = df1[predictors]
Y_train = df1['TARGET']
print (X_train.shape)
logistic = linear_model.LogisticRegression()

pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
n_components = [50, 60]
Cs = [0.1, 0.01]
estimator = grid_search.GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs))
estimator.fit(X_train, Y_train)

predictions = estimator.predict(df2[predictors])

submission = pd.DataFrame({"ID":df2["ID"],"TARGET":predictions})
submission = submission[['ID','TARGET']]
submission.to_csv('submission_PipeLine.csv')

