print("Testing started")
import numpy as np
from sklearn import linear_model

X = np.arange(0., 1., 0.001)
Y = X*20.+np.sin(X*10.*np.pi)/100.
X = X[:, np.newaxis]
print(len(X), len(Y))
regressor = linear_model.LinearRegression()
regressor.fit(X, Y)
print(regressor.score(X, Y))