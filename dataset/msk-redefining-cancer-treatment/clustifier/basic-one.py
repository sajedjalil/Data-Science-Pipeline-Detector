import numpy as np
import pandas as pd

from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("../input/training_variants")
test = pd.read_csv("../input/test_variants")

ID_train = train.ID
ID_test = test.ID

y = train.Class.values-1

train = train.drop(['ID','Class'], axis=1)
test = test.drop(['ID'], axis=1)

data = train.append(test)

X_data = pd.get_dummies(data).values

print(X_data.shape)

X = X_data[:train.shape[0]]
X_test = X_data[train.shape[0]:]

y_test = np.zeros((X_test.shape[0], max(y)+1))

n_folds = 5

kf = model_selection.StratifiedKFold(n_splits=n_folds, random_state=1, shuffle=True)

fold = 0
for train_index, test_index in kf.split(X, y):

	fold += 1

	X_train, X_valid    = X[train_index], 	X[test_index]
	y_train, y_valid    = y[train_index],   y[test_index]

	print("Fold", fold, X_train.shape, X_valid.shape)

	clf = LogisticRegression()

	clf.fit(X_train, y_train)

	p_train = clf.predict_proba(X_train)
	p_valid = clf.predict_proba(X_valid)
	p_test = clf.predict_proba(X_test)

	print(metrics.log_loss(y_train, p_train))
	print(metrics.log_loss(y_valid, p_valid))
	
	y_test += p_test/n_folds
	
classes = "class1,class2,class3,class4,class5,class6,class7,class8,class9".split(',')
subm = pd.DataFrame(y_test, columns=classes)
subm['ID'] = ID_test

subm.to_csv('basic_one.csv', index=False)	
	
	
	
	
