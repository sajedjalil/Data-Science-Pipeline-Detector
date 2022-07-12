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
from sklearn.model_selection import (
	train_test_split,
	StratifiedKFold,
	cross_val_score,
	GridSearchCV
)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.ensemble import (
	AdaBoostClassifier,
	GradientBoostingClassifier,
	RandomForestClassifier,
	ExtraTreesClassifier
)

# Read data
train_data = '../input/train.csv'
train_df = pd.read_csv(train_data)

# data transform
Y = train_df['species']
X = train_df.drop(['species', 'id'], axis=1)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

# encode Y
leaf_classes = Y.unique()
le = LabelEncoder()
le.fit(leaf_classes)
Y = le.transform(Y)

# create a validation dataset
validation_size = 0.3
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(
	X, Y, test_size=validation_size, random_state=seed)

# compare models
models = []
models.append(('LDA', LinearDiscriminantAnalysis(solver='svd')))
models.append(('SVM', SVC()))
models.append(('LR', LogisticRegression(solver='lbfgs', multi_class='multinomial')))
models.append(('ET', ExtraTreesClassifier()))

results = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=5, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	print('%s: %f (%f)'%(name, cv_results.mean(), cv_results.std()))

validation_results = []
for name, model in models:
	model.fit(X_train, Y_train)
	predictions = model.predict(X_validation)
	validation_results.append(accuracy_score(Y_validation, predictions))
	print('%s validation accuracy %f'%(name, validation_results[-1]))

# LR performs best
print('Doing grid search using LR...')
c_vals = [50, 100, 150]
tol = [0.0001, 0.001]
param_grid = {'C': c_vals, 'tol': tol}
model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
kfold = StratifiedKFold(n_splits=5, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold,
	n_jobs=-1, scoring='log_loss', refit=True)
grid_result = grid.fit(X, Y)

print('Best: %f using %s'%(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, std, parm in zip(means, stds, params):
	print('%f (%f) with: %r'%(mean, std, parm))

# Use best model for test prediction
print('Using best LR model to predict on test data...')
test_data  = '../input/test.csv'
test_df = pd.read_csv(test_data)

_id = test_df['id']

X = test_df.drop(['id'], axis=1)
X = scaler.transform(X)
predictions = grid.predict_proba(X)

submission = pd.DataFrame(predictions, index=_id, columns=le.classes_)
submission.to_csv('submission_LR.csv')
