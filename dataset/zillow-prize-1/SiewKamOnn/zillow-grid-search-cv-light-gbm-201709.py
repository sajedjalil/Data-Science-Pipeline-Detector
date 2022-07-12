
# ##filename=zillow_get_grid_search_cv_light_gbm_201709.py, edited on 21 Sep 2017 Thu 11:19 PM
# coding: utf-8

# In[1]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[34]:

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
import datetime as dt

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import KFold #for regressor
from sklearn.cross_validation import StratifiedKFold #  if the estimator is a classifier and y is either binary or multiclass

# from scipy.stats import randint, uniform
#
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout, BatchNormalization
# from keras.layers.advanced_activations import PReLU
# from keras.optimizers import Adam
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import Imputer


# In[5]:

pd.options.display.max_rows = 10


# In[20]:

#read data from disk
prop = pd.read_csv('../input/properties_2016.csv')
train = pd.read_csv("../input/train_2016_v2.csv")
# prop = pd.read_csv('input/properties_2016.csv')
# train = pd.read_csv("input/train_2016_v2.csv")


# In[21]:

# convert dtype

def convert_dtype(prop, from_type, to_type):
    for c, dtype in zip(prop.columns, prop.dtypes):
        if dtype == from_type:
            prop[c] = prop[c].astype(to_type)

convert_dtype(prop, np.float64, np.float32)
convert_dtype(train, np.float64, np.float32)


# In[22]:

df_train = train.merge(prop, how='left', on='parcelid')
df_train.fillna(df_train.median(),inplace = True)


# In[23]:

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
                         'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)


# In[24]:

train_columns = x_train.columns


# In[25]:

x_train.dtypes[x_train.dtypes == object].index.values


# In[12]:

x_train


# In[26]:

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    print(c)


# In[14]:

x_train['hashottuborspa']


# In[27]:

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)


# In[16]:

x_train


# In[28]:

x_train = x_train.values.astype(np.float32, copy=False)
d_train = lgb.Dataset(x_train, label=y_train)
x_train


# In[30]:

d_train


# In[47]:

# prepare for grid search

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, np.log(y_train))

    def predict(self, x):
        return np.exp(self.clf.predict(x))

    def fit(self,x,y):
        return self.clf.fit(x,y)

    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=np.log(y_train))
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        #np.exp because train converted label to np.log
        return np.exp(self.gbdt.predict(xgb.DMatrix(x)))


def get_oof(clf):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

#


# In[37]:

y_train


# In[46]:

seed = 19

np.random.seed(seed)
random.seed(seed)
n_folds=10

ntrain=train.shape[0]

cv = KFold(ntrain, n_folds=n_folds, shuffle=True, random_state=seed)


# In[49]:

import lightgbm as lgb


# In[58]:

# https://github.com/Microsoft/LightGBM/blob/master/docs/GPU-Performance.md
#  use 63 bins is better ?
#  use num_leaves = 255
#  max depth = 8
#  min_sum_hessian_in_leaf = 100
#  min_data_in_leaf =1
params_grid = {
#     'max_bin' : [10, 15, 20],
    'max_bin' : [15, 63, 255],
#     'learning_rate' : np.linspace(0.001, 0.009, 5), # shrinkage_rate
    'learning_rate' : np.linspace(0.05, 0.1, 3),
    'sub_feature' : np.linspace(0.1, 0.5, 3),     #0.345 feature_fraction (small values => use very different submodels),
    'bagging_fraction' : np.linspace(0.7, 0.9, 3), #0.85  sub_row,
    'bagging_freq' : np.linspace(30, 50, 3, dtype='int'),
#     'num_leaves' : np.linspace(400, 600, 3, dtype='int'), #512        # num_leaf,
#     'max_depth': [8, 9, 10],
#     'n_estimators': [5, 10, 25, 50],
}

params_fixed = {
    'objective': 'regression',
    'boosting_type' : 'gbdt',
    'bagging_seed' : 3,
    'feature_fraction_seed' : 2,
#     'min_hessian' : 0.05,     # min_sum_hessian_in_leaf,
    'min_sum_hessian_in_leaf' : 100,     # min_sum_hessian_in_leaf,
#     'min_data' : 500,         # min_data_in_leaf,
    'min_data_in_leaf' : 1,
    'num_leaves' : 255, #512        # num_leaf,
    'max_depth': 8,
    'silent': 1,
    'verbose' : 0,
    'metric' : 'l1',          # or 'mae',
}



# In[59]:
grid_search_cv_verbosity = 2

bst_grid = GridSearchCV(
    estimator=lgb.LGBMRegressor(**params_fixed, seed=seed),
    param_grid=params_grid,
    cv=cv,
    verbose = grid_search_cv_verbosity
    #scoring='accuracy'
)


# In[ ]:
print('Starting GridSearchCV ....')

bst_grid.fit(x_train, y_train)


# In[ ]:

print("Best accuracy obtained: {0}".format(bst_grid.best_score_))
print("Parameters:")
for key, value in bst_grid.best_params_.items():
    print("\t{}: {}".format(key, value))

