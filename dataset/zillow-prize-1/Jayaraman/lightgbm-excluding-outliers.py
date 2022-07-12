# -------------------------------------------------------------------------------- #

# OK, I"m going to try and tune this, but tuning is problematic for the following reasons:

# 1. In the current setup, choice of how many outliers to exclude will affect 
#    the contents of the validation set, so I'd have to do a bit of actual coding
#    if I want a validation set that I can use to tune outlier exclusion parameters
#    (and whatever set I came up with, there's no guarantee it would be a good
#    indicator of test set performance).

# 2. The original script, before excluding outliers, was already overfit to the
#    public test data, since it was forked from one that was selected for doing well
#    on the public leaderboard (although it was selected based on the old version
#    of the training data, so not quite as overfit as it used to be).  This means 
#    that choosing parameters based on public LB scores will bias the choice toward
#    excluding fewer outliers than we should.

# I'm not sure yet how to handle these issues, just playing it by ear

# Anyhow I'm probably going to do most of the tuning in public,
# and I welcome anyone else who wants to fork and particpate

# As I said in a comment, version 3 scored about 0.0649,
# considerably worse than the 0.06466 that the version including outliers got,
# but I'm hoping I can get to a tuned version that will at least score better
# than the unclean one.

# Version 4 is a shot in the dark with some parameter changes (outlier cutoff
# and validation split cutoff)

# And version 4 went over time, so I raised the learning rate and reduced num_boost_round

# -- Andy Harless  2017-07-03

# -------------------------------------------------------------------------------- #



import numpy as np
import pandas as pd
import lightgbm as lgb
import gc

print('Loading data ...')

train = pd.read_csv('../input/train_2016_v2.csv')
prop = pd.read_csv('../input/properties_2016.csv')

for c, dtype in zip(prop.columns, prop.dtypes):	
    if dtype == np.float64:		
        prop[c] = prop[c].astype(np.float32)

df_train = train.merge(prop, how='left', on='parcelid')

df_train=df_train[ df_train.logerror > -0.54 ]
df_train=df_train[ df_train.logerror < 0.54 ]
x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 72000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
x_train = x_train.values.astype(np.float32, copy=False)
x_valid = x_valid.values.astype(np.float32, copy=False)

d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_valid, label=y_valid)

params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0045 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'          # or 'mae'
params['sub_feature'] = 0.5      # feature_fraction 
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 30
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 350         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf

watchlist = [d_valid]
clf = lgb.train(params, d_train, 500, watchlist)

del d_train, d_valid; gc.collect()
del x_train, x_valid; gc.collect()

print("Prepare for the prediction ...")
sample = pd.read_csv('../input/sample_submission.csv')
sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')
del sample, prop; gc.collect()
x_test = df_test[train_columns]
del df_test; gc.collect()
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
x_test = x_test.values.astype(np.float32, copy=False)

print("Start prediction ...")
# num_threads > 1 will predict very slow in kernal
clf.reset_parameter({"num_threads":1})
p_test = clf.predict(x_test)

p_test = 0.97*p_test + 0.03*0.011

del x_test; gc.collect()

print("Start write result ...")
sub = pd.read_csv('../input/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

sub.to_csv('lgb_starter_2.csv', index=False, float_format='%.4f')