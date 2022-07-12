#
# linregbaseline.py
#
# This script establishes a benchmark for linear regression on
# the Prudential Life Insurance Assessment dataset. It makes use of minimal
# feature engineering, but demonstrates the importance of calibrating the
# regression model outputs. 
#
import re
import sys
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

np.random.seed(11)

#
# Quadratic Weighted Kappa objective
#
def kappa_weights(n):
    s = np.linspace(1, n, n)
    w = np.power(np.subtract.outer(s, s), 2.0) / np.power(np.subtract.outer(n, np.ones((n,))), 2)
    return w

def quadratic_weighted_kappa(a, b, n=8):
    
    a = np.clip(a.astype('int32') - 1, 0, n-1)
    b = np.clip(b.astype('int32') - 1, 0, n-1)
    
    sa = coo_matrix((np.ones(len(a)), (np.arange(len(a)), a)), shape=(len(a), n))
    sb = coo_matrix((np.ones(len(b)), (np.arange(len(b)), b)), shape=(len(a), n))
    
    O = (sa.T.dot(sb)).toarray()
    E = np.outer(sa.sum(axis=0), sb.sum(axis=0))
    E = np.divide(E, np.sum(E)) * O.sum()
    W = kappa_weights(n)
    
    return 1.0 - np.multiply(O, W).sum() / np.multiply(E, W).sum()

#
# Dataset definition
#
_ = categorical_vars = []
_.extend('Product_Info_%d' % d for d in [1,2,3,5,6,7])
_.extend('Employment_Info_%d' % d for d in [2,3,5])
_.extend('InsuredInfo_%d' % d for d in [1,2,3,4,5,6,7])
_.extend('Insurance_History_%d' %d for d in [1,2,3,4,7,8,9])
_.extend('Family_Hist_%d' % d for d in [1])
_.extend('Medical_History_%d' % d for d in 
	[2,3,4,5,6,7,8,9,11,12,13,14,16,17,18,19,20,21,22,23,25,26,
	27,28,29,30,31,33,34,35,36,37,38,39,40,41]
)

continuous_vars = re.split(r'\s*,\s*', '''Product_Info_4, Ins_Age, Ht, Wt,
BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6,
Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4,
Family_Hist_5''')

discrete_vars = re.split(r'\s*,\s*', '''Medical_History_1, Medical_History_10, Medical_History_15,
Medical_History_24, Medical_History_32''')

binary_vars = [ 'Medical_Keyword_%d' % (i + 1) for i in range(48) ]

#
# Dataset loading and encoding
#
data = pd.read_csv('../input/train.csv', index_col='Id')
train_instances = data.index
response_train = data.Response
del data['Response']

eval_data = pd.read_csv('../input/test.csv', index_col='Id')
test_instances = eval_data.index

encoded_data = pd.get_dummies(pd.concat([data, eval_data]), columns=categorical_vars)

X_train = encoded_data.filter(items=train_instances, axis=0)
X_train = X_train.fillna(X_train.mean())
y_train = response_train.values

X_eval = encoded_data.filter(items=test_instances, axis=0).fillna(X_train.mean())

#
#  Output warping optimization
#
from scipy.optimize import fmin_powell
from sklearn.base import BaseEstimator, TransformerMixin

def transform_output(y):
	return np.clip(np.round(y).astype('int32'), 1, 8)

class Calibrator(BaseEstimator, TransformerMixin):
		
	def _warp(self, x, b, a):
		return 1.0 + 7.0 /(1.0 + np.exp(-b-a*x))
		
	def fit(self, x, y):
		def obj(w):
			r = transform_output(self._warp(x, *w))
			return -quadratic_weighted_kappa(y, r)
		params0 = np.array([0.0, 1.0])
		self.params = fmin_powell(obj, params0)
		return self
		
	def transform(self, x):
		return self._warp(x, *self.params)
	
	def predict(self, x):
		return transform_output(self.transform(x))

#
# Linear Regression pipeline
#
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score

class LinearRegressionTransformer(LinearRegression, TransformerMixin):
	"""Extend LinearRegression so it can be used in non-final pipeline stage.
	"""	
	def transform(self, x):
		return self.predict(x)

p = Pipeline([
	('pca', PCA(n_components=256)),
	('reg', LinearRegressionTransformer()),
	('cal', Calibrator()),
])

score = cross_val_score(p, X_train.values, y_train,
	scoring=make_scorer(quadratic_weighted_kappa),
	n_jobs=-1, verbose=1
)

print("Cross-validation kappa", np.mean(score), np.std(score))

p.fit(X_train.values, y_train)
y_train_predict = p.predict(X_train.values)
response_train_predict = transform_output(y_train_predict)

print("Training kappa", quadratic_weighted_kappa(response_train, response_train_predict))

#
# Generate predictions for the test set
#
response_eval_predict = p.predict(X_eval.values)

pd.DataFrame({
	'Response': response_eval_predict
}, index=X_eval.index).to_csv('submission_file.csv')

#
# Plot the distribution of output predictions before and after calibration
#
import matplotlib.pyplot as plt

# Make another pipeline without the calibration step.
p2 = Pipeline(p.steps[:-1])

score = cross_val_score(p2, X_train.values, y_train,
	scoring=make_scorer(quadratic_weighted_kappa),
	n_jobs=-1, verbose=1
)
print("Cross-validation kappa (Uncalibrated)", np.mean(score), np.std(score))

plt.hist([
	transform_output(p2.predict(X_train.values)),
	y_train_predict,
	response_train,
], bins=np.linspace(0.5, 8.5, 9), label=['regression model', 'calibrated', 'true'])
plt.legend(loc="upper center")
plt.savefig('output_distribution.png')
