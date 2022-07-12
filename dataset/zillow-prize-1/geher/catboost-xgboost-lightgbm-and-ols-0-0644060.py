
# In[42]:

# Parameters
XGB_WEIGHT = 0.684
BASELINE_WEIGHT = 0.0056
CAT_WEIGHT = 0.1
OLS_WEIGHT = 0.0550

BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg

## geher v7 (best catboost ensemble so far)
# XGB 0.58
# CAT 0.08

## geher v7 
# XGB 0.58
# CAT 0.1

## geher v6
# XGB 0.48
# CAT 0.32

## geher v5 
# add catboost ensemble
# XGB_WEIGHT = 0.36
# CATBOOST_WEIGHT = 0.38

## geher v4
# num_boost_rounds = 250 based on version 41 from Andy
# weights at this point:
# XGB_WEIGHT = 0.63
# BASELINE_WEIGHT = 0.0056
# OLS_WEIGHT = 0.0550
# BASELINE_PRED = 0.0115

## geher v3
# change xgb to 0.63

## geher v2
# change xgb to 0.6315, increase outlier threshold to 0.419  

## geher v1
# change xgb weight from 0.6266 to 0.620 

# version 24
#    Revert to old BASELINE_WEIGHT and some cleanup

# version 21
#    Try BASELINE_WEIGHT = 0.0050, LB .0644125

# version 12
#    Try XGB_WEIGHT .6266, to do quadratic approximation, LB .0644123, already near-optimal

# version 11
#    Try XGB_WEIGHT .620 -> .6166, closer to old proportion, LB .0644133

# version 10
#    OLS_WEIGHT=.055, LB .0644127

# version 9
#    OLS_WEIGHT=.05, LB .0644129

# version 8
#    OLS_WEIGHT=.07, LB .0644136

# version 7
#    First attempt:  
#    OLS_WEIGHT=.06, LB .0644129
#    (XGB_WEIGHT=.62 to keep proportion with LGB roughly same as in old script)




# THE FOLLOWING SERIES OF COMMENTS REFERS TO VERSIONS OF
#    https://www.kaggle.com/aharless/xgb-w-o-outliers-lgb-with-outliers-combined
#    from which this script was forked

# version 61
#   Drop fireplacecnt and fireplaceflag, following Jayaraman:
#     https://www.kaggle.com/valadi/xgb-w-o-outliers-lgb-with-outliers-combo-tune5

# version 60
#   Try BASELINE_PRED=0.0115, since that's the actual baseline from
#     https://www.kaggle.com/aharless/oleg-s-original-better-baseline

# version 59
#   Looks like 0.0056 is the optimum BASELINE_WEIGHT

# versions 57, 58
#   Playing with BASELINE_WEIGHT parameter:
#     3 values will determine quadratic approximation of optimum

# version 55
#   OK, it doesn't get the same result, but I also get a different result
#     if I fork the earlier version and run it again.
#   So something weird is going on (maybe software upgrade??)
#   I'm just going to submit this version and make it my new benchmark.

# version 53
#   Re-parameterize ensemble (should get same result).

# version 51
#   Quadratic approximation based on last 3 submissions gives 0.3533
#     as optimal lgb_weight.  To be slightly conservative,
#     I'm rounding down to 0.35

# version 50
#   Quadratic approximation based on last 3 submissions gives 0.3073 
#     as optimal lgb_weight

# version 49
#   My latest quadratic approximation is concave, so I'm just taking
#     a shot in the dark with lgb_weight=.3

# version 45
#   Increase lgb_weight to 0.25 based on new quadratic approximation.
#   Based on scores for versions 41, 43, and 44, the optimum is 0.261
#     if I've done the calculations right.
#   I'm being conservative and only going 2/3 of the way there.
#   (FWIW my best guess is that even this will get a worse score,
#    but you gotta pay some attention to the math.)

# version 44
#   Increase lgb_weight to 0.23, per Nikunj's suggestion, even though
#     my quadratic approximation said I was already at the optimum

# verison 43
#   Higher lgb_weight, so I can do a quadratic approximation of the optimum

# version 42
#   The answer to the ultimate question of life, the universe, and everything
#     comes down to a slightly higher lgb_weight

# version 41
#   Trying Nikunj's suggestion of imputing missing values.

# version 39
#   Trying higher lgb_weight again but with old learning rate.
#   The new one did better with LGB only but makes the combination worse.

# version 38
#   OK back to baseline 0.2 weight

# version 37
#   Looks like increasing lgb_weight was better

# version 34
#   OK, try reducing lgb_weight instead

# version 32
#   Increase lgb_weight because LGB performance has imporved more than XGB
#   Increase learning rate for LGB: 0029 is compromise;  CV prefers 0033
#     (and reallly would prefer more boosting rounds with old value instead
#      but constaints on running time are getting hard)

# Version 27:
#   Control LightGBM's loquacity

# Version 26:
# Getting rid of the LightGBM validation, since this script doesn't use the result.
# Now use all training data to fit model.
# I have a separate script for validation:
#    https://www.kaggle.com/aharless/lightgbm-outliers-remaining-cv


import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import Pool, CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
import datetime as dt


# ## Read data

# ## LightGBM

# In[5]:

print( "\nReading data from disk ...")
prop = pd.read_csv('../input/properties_2016.csv')
train = pd.read_csv("../input/train_2016_v2.csv")

##### PROCESS DATA FOR LIGHTGBM

print( "\nProcessing data for LightGBM ..." )
for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.float64:
        prop[c] = prop[c].astype(np.float32)

df_train = train.merge(prop, how='left', on='parcelid')
df_train.fillna(df_train.median(),inplace = True)

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 
                         'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)


train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

x_train = x_train.values.astype(np.float32, copy=False)
d_train = lgb.Dataset(x_train, label=y_train)



##### RUN LIGHTGBM

params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'          # or 'mae'
params['sub_feature'] = 0.5      # feature_fraction -- OK, back to .5, but maybe later increase this
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params['verbose'] = 0

print("\nFitting LightGBM model ...")
clf = lgb.train(params, d_train, 430)

del d_train; gc.collect()
del x_train; gc.collect()

print("\nPrepare for LightGBM prediction ...")
print("   Read sample file ...")
sample = pd.read_csv("../input/sample_submission.csv")
print("   ...")
sample['parcelid'] = sample['ParcelId']
print("   Merge with property data ...")
df_test = sample.merge(prop, on='parcelid', how='left')
print("   ...")
del sample, prop; gc.collect()
print("   ...")
x_test = df_test[train_columns]
print("   ...")
del df_test; gc.collect()
print("   Preparing x_test...")
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
print("   ...")
x_test = x_test.values.astype(np.float32, copy=False)

print("\nStart LightGBM prediction ...")
p_test = clf.predict(x_test)

del x_test; gc.collect()

print( "\nUnadjusted LightGBM predictions:" )
print( pd.DataFrame(p_test).head() )


# ## XGBoost

# In[6]:

# This section is (I think) originally derived from Infinite Wing's script:
#   https://www.kaggle.com/infinitewing/xgboost-without-outliers-lb-0-06463
# inspired by this thread:
#   https://www.kaggle.com/c/zillow-prize-1/discussion/33710
# but the code has gone through a lot of changes since then


##### RE-READ PROPERTIES FILE
##### (I tried keeping a copy, but the program crashed.)

print( "\nRe-reading properties file ...")
prop = pd.read_csv('../input/properties_2016.csv')


##### PROCESS DATA FOR XGBOOST

print( "\nProcessing data for XGBoost ...")
for c in prop.columns:
    prop[c]=prop[c].fillna(-1)
    if prop[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(prop[c].values))
        prop[c] = lbl.transform(list(prop[c].values))

train_df = train.merge(prop, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
x_test = prop.drop(['parcelid'], axis=1)
# shape        
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

# drop out ouliers
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.419 ]
x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')     
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))



##### RUN XGBOOST

print("\nSetting up data for XGBoost ...")
# xgboost params
xgb_params = {
    'eta': 0.037,
    'max_depth': 5,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.8,   
    'alpha': 0.4, 
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

num_boost_rounds = 250
print("\nXGBoost tuned with CV in:")
print("   https://www.kaggle.com/aharless/xgboost-without-outliers-tweak ")
print("num_boost_rounds="+str(num_boost_rounds))

# train model
print( "\nTraining XGBoost ...")
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

print( "\nPredicting with XGBoost ...")
xgb_pred = model.predict(dtest)

print( "\nXGBoost predictions:" )
print( pd.DataFrame(xgb_pred).head() )

del train_df; gc.collect()
del x_train; gc.collect()
del x_test; gc.collect()
del prop; gc.collect()
del dtest; gc.collect()
del dtrain; gc.collect()


# # CatBoost

# In[10]:

print( "\nReading data from disk ...")
prop = pd.read_csv('../input/properties_2016.csv')
train = pd.read_csv("../input/train_2016_v2.csv")

##### PROCESS DATA FOR CATBOOST

print( "\nProcessing data for CatBoost ...")
for c in prop.columns:
    prop[c]=prop[c].fillna(-1)
    if prop[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(prop[c].values))
        prop[c] = lbl.transform(list(prop[c].values))

train_df = train.merge(prop, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
x_test = prop.drop(['parcelid'], axis=1)
# shape        
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

# drop outliers
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.419 ]
x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')     
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

##### RUN CATBOOST

print("\nFitting CatBoost model ...")
train_pool = Pool(x_train, y_train) # cat_features=[0,2,5])
test_pool = Pool(x_test) #, cat_features=[0,2,5]) 

model = CatBoostRegressor(rsm=0.8, depth=5, learning_rate=0.037, eval_metric='MAE')
#train the model
model.fit(train_pool)

# make the prediction using the resulting model
cat_preds = model.predict(test_pool)
print( pd.DataFrame(cat_preds[:10]).head() )

del x_train; gc.collect()
del train_pool; gc.collect()
del test_pool
del x_test; gc.collect()
del prop; gc.collect()


# ## OLS

# In[13]:

# This section is derived from the1owl's notebook:
#    https://www.kaggle.com/the1owl/primer-for-the-zillow-pred-approach
# which Andy Harless updated and made into a script:
#    https://www.kaggle.com/aharless/updated-script-version-of-the1owl-s-basic-ols

np.random.seed(17)
random.seed(17)

prop = pd.read_csv('../input/properties_2016.csv')
train = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])
submission = pd.read_csv("../input/sample_submission.csv")
print(len(train),len(prop),len(submission))

def get_features(df):
    df["transactiondate"] = pd.to_datetime(df["transactiondate"])
    df["transactiondate_year"] = df["transactiondate"].dt.year
    df["transactiondate_month"] = df["transactiondate"].dt.month
    df['transactiondate'] = df['transactiondate'].dt.quarter
    df = df.fillna(-1.0)
    return df

def MAE(y, ypred):
    return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)

train = pd.merge(train, prop, how='left', on='parcelid')
y = train['logerror'].values
test = pd.merge(submission, prop, how='left', left_on='ParcelId', right_on='parcelid')
prop = [] #memory

exc = [train.columns[c] for c in range(len(train.columns)) if train.dtypes[c] == 'O'] + ['logerror','parcelid']
col = [c for c in train.columns if c not in exc]

train = get_features(train[col])
test['transactiondate'] = '2016-01-01' #should use the most common training date
test = get_features(test[col])

reg = LinearRegression(n_jobs=-1)
reg.fit(train, y); print('fit...')
print(MAE(y, reg.predict(train)))
train = [];  y = [] #memory

test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
test_columns = ['201610','201611','201612','201710','201711','201712']


# ## Combine and save

# In[43]:

print( "\nCombining XGBoost, LightGBM, CatBoost and baseline predicitons ..." )
lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT - CAT_WEIGHT) / (1 - OLS_WEIGHT)
xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
baseline_weight0 =  BASELINE_WEIGHT / (1 - OLS_WEIGHT)
cat_weight0 = CAT_WEIGHT / (1 - OLS_WEIGHT)
pred0 = xgb_weight0*xgb_pred + baseline_weight0*BASELINE_PRED + lgb_weight*p_test + cat_weight0*cat_preds

print( "\nCombined XGB/LGB/CB/baseline predictions:" )
print( pd.DataFrame(pred0).head() )

print( "\nPredicting with OLS and combining with XGB/LGB/baseline predictions: ..." )
for i in range(len(test_dates)):
    test['transactiondate'] = test_dates[i]
    pred = OLS_WEIGHT*reg.predict(get_features(test)) + (1-OLS_WEIGHT)*pred0
    submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
    print('predict...', i)

print( "\nCombined XGB/LGB/CB/baseline/OLS predictions:" )
print( submission.head() )


##### WRITE THE RESULTS

from datetime import datetime

print( "\nWriting results to disk ..." )
submission.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print( "\nSaved ...")


