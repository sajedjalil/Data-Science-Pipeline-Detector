import numpy as np
import pandas as pd
import xgboost as xgb
import gc

from sklearn.ensemble import RandomForestClassifier
from hep_ml.losses import BinFlatnessLossFunction
from hep_ml.gradientboosting import UGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

print('Loading data ...')

train = pd.read_csv('../input/train_2016.csv')
prop = pd.read_csv('../input/properties_2016.csv')
sample = pd.read_csv('../input/sample_submission.csv')

print('Binding to float32')

for c, dtype in zip(prop.columns, prop.dtypes):
	if dtype == np.float64:
		prop[c] = prop[c].astype(np.float32)

print('Creating training set ...')

df_train = train.merge(prop, how='left', on='parcelid')

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 80000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

print('Building DMatrix...')

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

del x_train, x_valid; gc.collect()

# to adapt
#features = list(train.columns[1:-5])
#print("Train a UGradientBoostingClassifier")
#loss = BinFlatnessLossFunction(['mass'], n_bins=15, uniform_label=0)
#clf = UGradientBoostingClassifier(loss=loss, n_estimators=150, subsample=0.1, # n_estimators = 75
#                                  max_depth=7, min_samples_leaf=10,
#                                  learning_rate=0.1, train_features=features, random_state=11)
#clf.fit(train[features + ['mass']], train['signal'])
#fb_preds = clf.predict_proba(test[features])[:,1]
#print("Train a Random Forest model")
#rf = RandomForestClassifier(n_estimators=375, n_jobs=-1, criterion="entropy", random_state=1)
#rf.fit(train[features], train["signal"]) # used to be n_estimators=300, 375 is better, 250 could be fine

#print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.2,# used to be 0.2 or 0.1
          "max_depth": 7, # used to be 5 or 6
          "min_child_weight": 1,
          "silent": 1,
          "colsample_bytree": 0.7,
          "seed": 1}
#num_trees=450 #used to be 300, 375 is better
#gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)

#print("Make predictions on the test set")
# test_probs = (0.35*rf.predict_proba(test[features])[:,1]) + (0.35*gbm.predict(xgb.DMatrix(test[features])))+(0.15*predskeras) + (0.15*fb_preds) 
#test_probs = (0.24*rf.predict_proba(test[features])[:,1]) + (0.3*gbm.predict(xgb.DMatrix(test[features])))+(0.26*predskeras) + (0.20*fb_preds) #is better


print('Training ...')

params = {}
params['eta'] = 0.05
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 5
params['silent'] = 1
params['colsample_bytree'] = 0.9
params['min_child_weight'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

del d_train, d_valid

print('Building test set ...')

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')

del prop; gc.collect()

x_test = df_test[train_columns]
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

del df_test, sample; gc.collect()

d_test = xgb.DMatrix(x_test)

del x_test; gc.collect()

print('Predicting on test ...')

p_test = clf.predict(d_test)

del d_test; gc.collect()

sub = pd.read_csv('../input/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

print('Writing csv ...')
sub.to_csv('xgb_starter.csv', index=False, float_format='%.4f') # Thanks to @inversion