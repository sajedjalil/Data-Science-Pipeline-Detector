# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy
import pandas as pd
from pandas import read_csv
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, confusion_matrix, recall_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dropout, BatchNormalization
import glob, re
import numpy as np
import pandas as pd
from sklearn import *
from datetime import datetime
import random

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from scipy.stats import randint
import lightgbm as lgb
from scipy.stats import uniform
import xgboost as xgb
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# Data Wrangling courtesy of https://www.kaggle.com/the1owl/surprise-me
import glob, re
import numpy as np
import pandas as pd
from sklearn import *
from datetime import datetime

data = {
    'tra': pd.read_csv('../input/air_visit_data.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/air_reserve.csv'),
    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    'id': pd.read_csv('../input/store_id_relation.csv'),
    'tes': pd.read_csv('../input/sample_submission.csv'),
    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

for df in ['ar','hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors':'rv1'})
    tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors':'rv2'})
    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)

#sure it can be compressed...
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 

stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 
lbl = preprocessing.LabelEncoder()
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 

train = pd.merge(train, stores, how='left', on=['air_store_id','dow']) 
test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])

for df in ['ar','hr']:
    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 
    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])
    

#train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)

#train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
#train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
#train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2

#test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
#test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
#test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2

col = [c for c in train if c not in ['id', 'air_store_id','visit_date','visitors']]
train = train.fillna(-1)
test = test.fillna(-1)

# XGB starter template borrowed from @anokas
# https://www.kaggle.com/anokas/simple-xgboost-starter-0-0655

print('Binding to float32')

for c, dtype in zip(train.columns, train.dtypes):
    if dtype == np.float64:
        train[c] = train[c].astype(np.float32)
        
for c, dtype in zip(test.columns, test.dtypes):
    if dtype == np.float64:
        test[c] = test[c].astype(np.float32)

print(train.head())
print(test.info())

y = np.log1p(train['visitors'].values)

# Remove the Y value from our training X set
xtrain = train.drop(['visitors', 'air_store_id','visit_date'], axis = 1)
#print(xtrain.shape, y.shape)

test1 = test.drop(['visitors', 'air_store_id', 'id','visit_date'], axis=1)
#test = test.drop(['visitors'], axis=1)
#train.set_index("visit_date", inplace=True)
#test1.set_index("visit_date", inplace=True)
print(train.info())
print(test1.info())

xtrain = train.sort_values('visit_date')#
xtrain.set_index("visit_date", inplace=True)#
y = np.log1p(train['visitors'].values)#

# Remove the Y value from our training X set
xtrain = xtrain.drop(['visitors', 'air_store_id'], axis = 1)#

from sklearn.cross_validation import train_test_split#

X_train, X_test, y_train, y_test = train_test_split = \
    train_test_split(xtrain, y, test_size=0.3, random_state=42)

# Need to scale the training and the test data seperately or you will get leakage from test to training y values.
values_y_train = y_train.reshape((len(y_train), 1))
scaler_y_train = MinMaxScaler(feature_range=(0, 1))
scaled_y_train = scaler_y_train.fit_transform(values_y_train)

# normalize features
scaler_x_train = MinMaxScaler(feature_range=(0, 1))
scaled_x_train = scaler_x_train.fit_transform(X_train)

def RMSLE(y, pred):
    return mean_squared_error(y, pred)**0.5


# In[38]:

seed = 7
num_trees = 200

KNN = KNeighborsRegressor(algorithm='auto', leaf_size=76, metric='l1',
        metric_params=None, n_jobs=1, n_neighbors=95, p=1, weights='uniform')
rf_reg = RandomForestRegressor(n_estimators = num_trees, random_state=seed)
gb_reg = GradientBoostingRegressor(n_estimators = num_trees, random_state=seed,
        learning_rate = 0.1, max_depth = 8 )
xtra_reg = ExtraTreesRegressor(n_estimators = num_trees, random_state=seed)
ada_reg = AdaBoostRegressor(n_estimators = num_trees, random_state=seed)

# prepare models
models = []
models.append(('GBR', gb_reg))
models.append(('RF', rf_reg ))
models.append(('KNN', KNN))
models.append(('ETR', xtra_reg))
models.append(('ADA', ada_reg))


results = []
names = []
scoring = 'neg_mean_squared_error'
for name, model in models:
    kfold = KFold(n_splits=2, random_state=7)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
'''
# Now let's create a stacking regressor of the top 3.
#https://rasbt.github.io/mlxtend/user_guide/regressor/StackingRegressor/
from mlxtend.regressor import StackingRegressor

streg = StackingRegressor(regressors=[KNN, rf_reg], 
                           meta_regressor=gb_reg)

streg.fit(xtrain, y)
pred_final =streg.predict(test1)

y_pred = np.expm1(pred_final)
print(y_pred)

test['visitors'] = y_pred

test[['id','visitors']].to_csv('stack_submission.csv', index=False, float_format='%.3f')                        

RMSE
849.1s
11
GBR: 0.576630 (0.016656)
1551.9s
12
RF: 0.518752 (0.013106)
2080.0s
13
KNN: 0.564786 (0.017890)

Tuned
844.5s
11
GBR: 0.596233 (0.015767)
1534.8s
12
RF: 0.589292 (0.013938)
2302.1s
13
KNN: 0.572851 (0.013516)


'''