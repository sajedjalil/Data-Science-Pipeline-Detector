# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import numpy as np
import pandas as pd

from patsy import dmatrix
from sklearn.linear_model import LinearRegression, ElasticNetCV, RidgeCV

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
cat_cols = [isinstance(c, str) for c in train.ix[0]]
train = pd.concat((train.loc[:, np.logical_not(cat_cols)], pd.get_dummies(train.loc[:, cat_cols])), axis=1)
train.reindex(np.random.permutation(train.index))
X_train = train.drop(['Id', 'Hazard'], axis=1).values
y_train = train['Hazard'].values

idxs = int(0.3*X_train.shape[0])
print(idxs)
X_val = X_train[:idxs]
y_val = y_train[:idxs]
X_train = X_train[idxs:]
y_train = y_train[idxs:]

test  = pd.read_csv("../input/test.csv")
cat_cols = [isinstance(c, str) for c in test.ix[0]]
test = pd.concat((test.loc[:, np.logical_not(cat_cols)], pd.get_dummies(test.loc[:, cat_cols])), axis=1)
X_test = test.drop(['Id'], axis=1)

linreg = LinearRegression()
linreg.fit(X_train, y_train)
print('R squared (Linear Regression): {:.4f}'.format(linreg.score(X_val, y_val)))

enet = ElasticNetCV(l1_ratio=np.linspace(0.5, 1),
                    verbose=True)
enet.fit(X_train, y_train)
print('R squared (Elastic Net): {:.4f}'.format(enet.score(X_val, y_val)))

ridge = RidgeCV(alphas=np.linspace(1e-7, 1))
ridge.fit(X_train, y_train)
print('R squared (Ridge): {:.4f}'.format(ridge.score(X_val, y_val)))

output = pd.DataFrame(columns=['Id', 'Hazard'])
output['Hazard'] = ridge.predict(X_test)
output['Id'] = test['Id'].values.astype(int)
output.to_csv('output.csv', index=False)