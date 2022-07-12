__author__ = 'rain'
 
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
import pandas as pd
X_train = pd.DataFrame({"x" : [1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 8, 10]})
y_train= pd.DataFrame({"y" : [-890, -1411, -1560, -2220, -2091, -2878, -3537, -3268, -3920, -4163, -5471, -5157]})
clf_linear = LinearRegression()
clf_linear.fit(X_train, y_train)
print(clf_linear.coef_)
print(clf_linear.intercept_)