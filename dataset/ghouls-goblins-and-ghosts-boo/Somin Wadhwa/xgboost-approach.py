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
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

#ML Imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

#importing data
train_DF = pd.read_csv('../input/train.csv')
test_DF = pd.read_csv('../input/test.csv')

indexes = test_DF['id']
train_DF.drop(['id'], axis = 1, inplace = True)
test_DF.drop(['id'], axis = 1, inplace = True)

figure, (ax1,ax2) = plt.subplots(1,2, figsize = (15,5))
sns.countplot(x = 'type', data = train_DF, ax = ax1)
sns.countplot(x = 'type', hue = 'color', data = train_DF, ax = ax2)

corr = train_DF.corr()
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap)

Y = train_DF['type']
train_DF.drop(['type'], axis = 1, inplace = True)
train_DF = pd.get_dummies(train_DF)
test_DF = pd.get_dummies(test_DF)
X = train_DF
X_test = test_DF
leY = LabelEncoder()
Y = leY.fit_transform(Y)

X,Xcv,Y,Ycv = train_test_split(X,Y, test_size = 0.20, random_state = 36)
print (X.shape)
print(Y.shape)

logreg = LogisticRegression()

parameter_grid = {'solver' : ['newton-cg', 'lbfgs'],
                  'multi_class' : ['multinomial'],
                  'C' : [0.005, 0.01, 1, 10],
                  'tol': [0.0001, 0.001, 0.005, 0.01]
                 }

grid_search_logit = GridSearchCV(logreg, param_grid=parameter_grid, cv=StratifiedKFold(3))
grid_search_logit.fit(X, Y)

print('Best score: {}'.format(grid_search_logit.best_score_))
print('Best parameters: {}'.format(grid_search_logit.best_params_))

Ycv2 = grid_search_logit.predict(Xcv)
accuracy_score(Ycv,Ycv2)

import xgboost as xgb
dtrain = xgb.DMatrix(X, label = Y)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":3, "eta":0.0001, 'nthread':6, 'gamma':0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()

model = xgb.XGBClassifier(learning_rate=0.0001, n_estimators=360, max_depth=3, nthread=6, gamma=0.1)
model.fit(X,Y)

model.score(X,Y)

Y_pred = model.predict(X_test)
pred = leY.inverse_transform(Y_pred)

submission = pd.DataFrame({'id':indexes,
                           'type':pred})
submission.to_csv('submission.csv', index=False)