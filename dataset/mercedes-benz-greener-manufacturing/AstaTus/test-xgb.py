# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
from scipy.stats import skew

import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn import metrics
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.grid_search import GridSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv', index_col=0)
test_df = pd.read_csv('../input/test.csv', index_col=0)

label_df = pd.DataFrame(train_df['y'])
label_df = np.log1p(label_df)
train_df.drop(['y'], axis=1, inplace=True)

def munge(df):
    all_df = pd.DataFrame(df.values, index=df.index, columns=df.columns, copy=True)
    all_df.drop(['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8'], axis=1, inplace=True)
    
    
    #删除取值相同的特征
    all_df.drop(['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290' ,'X293' ,'X297', 'X330' ,'X347'], axis=1, inplace=True)
    
    #构造新特征
    all_df['parts'] = all_df.sum(axis=1)
    return all_df

munged_train_df = munge(train_df)
munged_test_df = munge(test_df)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(pd.DataFrame(munged_train_df['parts']))

scaled = scaler.transform(pd.DataFrame(munged_train_df['parts']))
munged_train_df['parts'] = scaled

scaled = scaler.transform(pd.DataFrame(munged_test_df['parts']))
munged_test_df['parts'] = scaled

# Convert categorical features using one-hot encoding.
def onehot(onehot_df, df, column_name, fill_na):
    onehot_df[column_name] = df[column_name]
    if fill_na is not None:
        onehot_df[column_name].fillna(fill_na, inplace=True)

    dummies = pd.get_dummies(onehot_df[column_name], prefix = column_name)
    
    onehot_df = onehot_df.join(dummies)
    onehot_df = onehot_df.drop([column_name], axis=1)
    return onehot_df

def munge_onehot(df):
    onehot_df = pd.DataFrame(index = df.index)

    onehot_df = onehot(onehot_df, df, "X0", None)
    onehot_df = onehot(onehot_df, df, "X1", None)
    onehot_df = onehot(onehot_df, df, "X2", None)
    onehot_df = onehot(onehot_df, df, "X3", None)
    onehot_df = onehot(onehot_df, df, "X4", None)
    onehot_df = onehot(onehot_df, df, "X5", None)
    onehot_df = onehot(onehot_df, df, "X6", None)
    onehot_df = onehot(onehot_df, df, "X8", None)
    
    return onehot_df

onehot_df = munge_onehot(train_df)
munged_train_df = munged_train_df.join(onehot_df)

onehot_df = munge_onehot(test_df)
munged_test_df = munged_test_df.join(onehot_df)

#删除test中有的  而train中没有的
munged_test_df.drop(['X0_ae', 'X0_ag', 'X0_an', 'X0_av', 'X0_bb', 'X0_p',
                     'X2_ab', 'X2_ad', 'X2_aj', 'X2_ax', 'X2_u', 'X2_w', 'X5_a', 'X5_b', 'X5_t', 'X5_z'], axis=1, inplace=True)

#删除train中有的  而test中没有的
munged_train_df.drop(['X0_aa', 'X0_ab', 'X0_ac', 'X0_q', 'X2_aa', 'X2_ar', 'X2_c', 'X2_l', 'X2_o', 'X5_u'], axis=1, inplace=True)

s = munged_train_df.shape[0]
drop_names = []
for c in munged_train_df.drop(['parts'], axis=1).columns:
    a = munged_train_df[c].value_counts()[0] / s
    b = munged_train_df[c].value_counts()[1] / s
    if (a < 0.05 or b < 0.05):
        print('%s p1 = %f p2 = %f'%(c, a, b))
        drop_names.append(c)
        
munged_train_df.drop(drop_names, axis=1, inplace=True)
munged_test_df.drop(drop_names, axis=1, inplace=True)

from sklearn.decomposition import PCA, FastICA

n_comp = 12

pca = PCA(n_components=n_comp, random_state=42)
pca2_results_train = pca.fit_transform(munged_train_df)
pca2_results_test = pca.transform(munged_test_df)

#ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(munged_train_df)
ica2_results_test = ica.transform(munged_test_df)

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    munged_train_df['pca_' + str(i)] = pca2_results_train[:,i-1]
    munged_test_df['pca_' + str(i)] = pca2_results_test[:, i-1]
    
    munged_train_df['ica_' + str(i)] = ica2_results_train[:,i-1]
    munged_test_df['ica_' + str(i)] = ica2_results_test[:, i-1]
    
import xgboost as xgb

y_mean = np.mean(label_df['y'])
# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 100, 
    'eta': 0.1,
    'max_depth': 3,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(munged_train_df.values, label_df['y'].values)
dtest = xgb.DMatrix(munged_test_df.values)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=500, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=50, 
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print(num_boost_rounds)

# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

# check f2-score (to get higher score - increase num_boost_round in previous cell)
from sklearn.metrics import r2_score

# now fixed, correct calculation
print(r2_score(dtrain.get_label(), model.predict(dtrain)))

# make predictions and save results
y_pred = model.predict(dtest)
y_pred = np.expm1(y_pred)
output = pd.DataFrame({'id': munged_test_df.index, 'y': y_pred})
output.to_csv('xgboost-depth{}-pca-ica.csv'.format(xgb_params['max_depth']), index=False)