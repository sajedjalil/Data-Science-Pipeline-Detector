import xgboost as xgb
import numpy as np
from sklearn.datasets import load_boston, load_iris

boston = load_boston()
X = boston.data
y = boston.target

y[:250] = 0
y[250:] = 1

model = xgb.XGBClassifier()
print(model.fit(X, y))
"""
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.datasets import load_boston, load_iris
from sklearn.cross_validation import cross_val_predict, cross_val_score
def accuracy(y, y_hat):
	return np.mean(y == y_hat)
def rmspe(y, y_hat):
    i = y != 0
    return np.sqrt(np.mean(np.power((y[i] - y_hat[i]) / y[i].astype(float), 2)))
    
#-------------------------------------------------------------------------------

boston = load_boston()
X = boston.data
y = boston.target

#-------------------------------------------------------------------------------

print('Regression (boston)\n')

model = LinearRegression()
y_hat = cross_val_predict(model, X, y, cv = 10)
print('LinearRegression:', rmspe(y, y_hat))

model = GradientBoostingRegressor()
y_hat = cross_val_predict(model, X, y, cv = 10)
print('GradientBoostingRegressor:', rmspe(y, y_hat))

model = xgb.XGBRegressor()
y_hat = cross_val_predict(model, X, y, cv = 10)
print('xgb.XGBRegressor:', rmspe(y, y_hat))

#-------------------------------------------------------------------------------

iris = load_iris()
X = iris.data
y = iris.target

#-------------------------------------------------------------------------------

print('\nClassification (iris)\n')

model = LogisticRegression()
y_hat = cross_val_predict(model, X, y, cv = 10)
print('LogisticRegression:', accuracy(y, y_hat))

model = xgb.XGBClassifier()
y_hat = cross_val_predict(model, X, y, cv = 10)
print('xgb.XGBClassifier:', accuracy(y, y_hat))

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
"""