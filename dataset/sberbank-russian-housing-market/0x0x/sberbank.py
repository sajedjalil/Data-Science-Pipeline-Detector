# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))


# Any results you write to the current directory are saved as output.
train_ds = pd.read_csv("../input/train.csv")
test_ds = pd.read_csv("../input/test.csv")
macro_ds = pd.read_csv("../input/macro.csv")

train_ds = pd.merge_ordered(train_ds, macro_ds, on='timestamp', how='left')
test_ds = pd.merge_ordered(test_ds, macro_ds, on='timestamp', how='left')

# Data cleaning
print(train_ds.shape)

# We have some properties with full_sq=1 ==> We will delete this entries
train_ds = train_ds[train_ds['full_sq'] > 1]
print(train_ds.shape)

# Alse we have some properties with full_sq < life_sq. That looks strange
tmp_s = train_ds[train_ds.life_sq > 1]

kk = tmp_s['life_sq']/tmp_s['full_sq']
print(np.mean(kk))

for i, row in train_ds.iterrows():
    if(np.isnan(row['life_sq']) or row['life_sq'] <= 1):
        train_ds.set_value(i, 'life_sq', row['full_sq'] * np.mean(kk)) 

train_ds = train_ds[train_ds['life_sq'] < train_ds['full_sq']]
print(train_ds.shape)

# We have some wrong values for build_year
tmp_s = tmp_s[(tmp_s['build_year'] > 500) & (tmp_s['build_year'] < 3000)]

print(tmp_s['build_year'].mean())

train_ds['build_year'].fillna(tmp_s['build_year'].mean(), inplace=True)
train_ds = train_ds[train_ds['build_year'] > 500]
train_ds = train_ds[train_ds['build_year'] < 3000]
print(train_ds.shape)

ls = list()
for i in train_ds.columns:
    if train_ds[i].dtype=='object':
        ls.append(train_ds[i].name)

ls.remove('timestamp')
for i, val in enumerate(ls):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_ds[val].values)) 
    train_ds[val] = lbl.transform(list(train_ds[val].values)) 

missing_ds = train_ds.isnull().sum(axis=0).reset_index()
missing_ds.columns = ['column_name', 'missing_count']
missing_ds = missing_ds.ix[missing_ds['missing_count']>0]
missing_ds = missing_ds.sort_values(['missing_count'], ascending=[True])
ind = np.arange(missing_ds.shape[0])
    
for i, row in missing_ds.iterrows():
    train_ds[row['column_name']].fillna(train_ds[row['column_name']].median(), inplace = True)

train_ds['state'].fillna(0, inplace=True)
train_ds['material'].fillna(0, inplace=True)
train_ds['build_count_mix'].fillna(9, inplace=True)
train_ds['build_count_foam'].fillna(3, inplace=True)

# Split dataframe
y_train = train_ds['price_doc'].values


#clean_train_ds.drop(["id", "timestamp", "price_doc"], axis=1)
train_ds = train_ds.drop(["id", "price_doc", "timestamp"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(train_ds, y_train, test_size = 0.7, random_state=42)
    
print(x_train.shape[0])
print(x_test.shape[0])

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 1.0,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train, feature_names = train_ds.columns)
dtest = xgb.DMatrix(x_test, feature_names = train_ds.columns)


# Uncomment to tune XGB `num_boost_rounds`
model = xgb.train(xgb_params, dtrain, num_boost_round=1000)

y_xgb = model.predict(dtest)

#df_sub = pd.DataFrame({'price_doc': y_xgb})

#print("RMSLE XGB: {:0.3f}".format(rmsle(y_test, df_sub)))


importances = model.get_fscore()

importance_frame = pd.DataFrame({'Importance': list(importances.values()), 'Feature': list(importances.keys())})
importance_frame.sort_values(by = 'Importance', ascending=False, inplace = True)

print(importance_frame.head(50))

#importance_frame.to_csv('sub.csv', index=False)

#e1 = df_sub['price_doc'].values
#print(e1)
#print(y_test)

#print("XGB: %.5f" % accuracy_score(y_test, e1))

#clf = MultinomialNB()
#clf.fit(x_train, y_train)

#rf = RandomForestClassifier(n_estimators = 10, n_jobs=1)
#rf.fit(x_train, y_train)


    
    
    
    