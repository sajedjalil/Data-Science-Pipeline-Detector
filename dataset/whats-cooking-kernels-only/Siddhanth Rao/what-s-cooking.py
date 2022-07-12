# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import h2o
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
#from sklearn import cross_validation, metrics   #Additional scklearn functions
#from sklearn.grid_search import GridSearchCV   #Performing grid search
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
with open('../input/train.json', 'r') as f:
    data = json.load(f)
train = pd.DataFrame(data)
with open('../input/test.json', 'r') as f:
    data2 = json.load(f)
test = pd.DataFrame(data2)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
mlb = MultiLabelBinarizer()

df = pd.DataFrame(mlb.fit_transform(train['ingredients']),columns=mlb.classes_, index=train.index)
train = pd.merge(train, df, left_index=True, right_index=True)

df2 = pd.DataFrame(mlb.fit_transform(test['ingredients']),columns=mlb.classes_, index=test.index)
test = pd.merge(test, df2, left_index=True, right_index=True)

X = train.drop(['cuisine','id','ingredients'],axis=1)
Y = train[['cuisine']]


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=12)

# Grid search cross validation
#from sklearn.model_selection import GridSearchCV
#from sklearn.linear_model import LogisticRegression
#grid={"C":np.logspace(-3,3,7,10,12), "penalty":["l1","l2"]}# l1 lasso l2 ridge
#logreg=LogisticRegression()
#logreg_cv=GridSearchCV(logreg,grid,cv=10)
#logreg_cv.fit(X_train,y_train)

#print("tuned hyperparameters :(best parameters) ",logreg_cv.best_params_)
#print("accuracy :",logreg_cv.best_score_)

#LogisticRegression

from sklearn.linear_model import LogisticRegression

clf1 = LogisticRegression(penalty='l2',C=1,dual=False,solver='lbfgs',multi_class='multinomial')
clf1.fit(X_train , y_train)
clf1.score(X_test, y_test)

# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_test, y_test)

sub = pd.DataFrame({'id': id, 'cuisine': Y_pred}, columns=['id', 'cuisine'])
sub.to_csv('sd_output.csv', index=False)