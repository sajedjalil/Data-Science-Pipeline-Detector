

### Thanks for Mr.Alexandru Papiu and his grate kernel.
### https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

train["SalePrice"] = np.log1p(train["SalePrice"])

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)

all_data = all_data.fillna(all_data.mean())

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice


#### models selection

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.0004)

model = lasso

### prediction
model.fit(X_train, y)

preds = np.expm1(model.predict(X_test))
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution.to_csv("lasso.csv", index = False)









