# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score, log_loss
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
import category_encoders as ce
from numpy.testing import assert_almost_equal
from functools import reduce
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


## KAGGLE bioresponse https://www.kaggle.com/c/bioresponse#Evaluation

train_url = '../input/train.csv'
test_url = '../input/test.csv' ## ignoring-- it doesn't have 'Activity' 

df_ = pd.read_csv(train_url)
df_test = pd.read_csv(test_url) # doesn't have 'Activity'
assert all([x==0 for x in df_.isna().sum().values])
assert all([pd.api.types.is_numeric_dtype(df_[feat]) for feat in df_.columns])
dependent='Activity'

X_train, X_test, y_train, y_test = train_test_split(df_.drop(dependent, axis=1), 
                                                    df_[dependent], 
                                                    train_size=0.8, test_size=0.2)

#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X=df_.drop(dependent, axis=1)
y=df_[dependent]

pcan = 218
l_alpha = 0.15848931924611143

transformer = FunctionTransformer(np.log1p, validate=True)
scl = StandardScaler(with_std=False)
logistic = SGDClassifier(alpha=l_alpha, max_iter=10000, tol=np.exp(-5), random_state=0,
                         loss='modified_huber', penalty='l2')
pca = PCA(n_components=pcan)

pipe = Pipeline(steps=[('standardize', scl), ('logarithm', transformer), 
                       ('pca', pca), ('logistic', logistic)])

pipe.fit(X_train, y_train)
y_pred = pipe.predict_proba(df_test)

print(f'My Submission is {y_pred}')