# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train_df = pd.read_csv("../input/train.csv")
train_df.shape

test_df = pd.read_csv("../input/test.csv")
test_df.shape


# process columns, apply LabelEncoder to categorical features for training set
for c in train_df.columns:
    if train_df[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train_df[c].values)) 
        train_df[c] = lbl.transform(list(train_df[c].values))
        
train_df.shape


# process columns, apply LabelEncoder to categorical features for test set
for c in test_df.columns:
    if test_df[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(test_df[c].values)) 
        test_df[c] = lbl.transform(list(test_df[c].values))
        
test_df.shape

# Add decomposed components: PCA / ICA etc.

train = train_df
test = test_df

from sklearn.decomposition import PCA, FastICA
n_comp = 10

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

# xgboost

# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 500, 
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=500, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=50, 
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
#print(num_boost_rounds)

# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

# check f2-score
print(r2_score(dtrain.get_label(), model.predict(dtrain)))

# make predictions and save results
y_pred = model.predict(dtest)
output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('xgboost-depth{}-pca-ica.csv'.format(xgb_params['max_depth']), index=False)


