# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

test["y"] = None

full = train.append(test)

full_objects = full.loc[:, full.dtypes == 'object']
y = full.y
full_objects = full_objects.drop("y",axis = 1)

object_columns = full_objects.columns

#dummies = pd.get_dummies(full_objects)
#print(dummies.head())

lbl = LabelEncoder()

full_objects = full_objects.apply(lbl.fit_transform)


full[object_columns] = full_objects

#full = full.drop(object_columns,axis = 1)
#full[dummies.columns] = dummies
#full= full.T.drop_duplicates().T
print(full.shape)

train = full[:len(train)]
test = full[len(train):len(train) + len(test)]

print(train.shape)
print(test.shape)

train = train.drop("ID",axis = 1) 
test_ID = test.ID
test = test.drop("ID", axis = 1)
test = test.drop("y", axis = 1)

y_train = train.y
y_mean = np.mean(y_train)

from sklearn.decomposition import PCA, FastICA
n_comp = 11

# PCA
pca = PCA(n_components=n_comp, random_state=42)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    train['pca_' + str(i)] = pca2_results_train[:,i-1]
    test['pca_' + str(i)] = pca2_results_test[:, i-1]
    
    train['ica_' + str(i)] = ica2_results_train[:,i-1]
    test['ica_' + str(i)] = ica2_results_test[:, i-1]
    
y_train = train["y"]
y_mean = np.mean(y_train)

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

xgb_params = {
    'n_estimators': 708, 
    'learning_rate': 0.005,
    'max_depth': 4,
    'subsample': 0.95,
    'booster':'dart',
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, 
    'n_jobs':2,
    'rate_drop': 0.1,
     'skip_drop': 0.5,
    'silent': 1
}

dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)

# xgboost, cross-validation
import scipy.stats as st

# specify parameters and distributions to sample from
"""
param_dist = {"n_estimators": sp_randint(500, 1000),
              "learning_rate": st.expon(scale=0.01),
              "reg_lambda":st.expon(scale=10),
              "reg_alpha":st.expon(scale=0.1)
              }

# run randomized search
clf = xgb.XGBRegressor(** xgb_params)
n_iter_search = 10
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search,verbose=5)


random_search.fit(train.drop('y',axis=1), y_train)         
"""
"""
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=1000, 
                   early_stopping_rounds=50,
                   verbose_eval=50, 
                   show_stdv=False
                  )
"""
#num_boost_rounds = len(cv_result)
#print(num_boost_rounds)

# train model
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=1000)
score = model.eval(dtrain)
print(score)
#print(random_search.best_params_)
#print(random_search.best_score_)
#y_pred = random_search.predict(test)
y_pred = model.predict(dtest)
output = pd.DataFrame({'id': test_ID, 'y': y_pred})
output.to_csv('submission.csv', index=False)




