# Parameters
FUDGE_FACTOR = 1.1200  # Multiply forecasts by this

XGB_WEIGHT = 0.6200
BASELINE_WEIGHT = 0.0100
OLS_WEIGHT = 0.0620
NN_WEIGHT = 0.0800

XGB1_WEIGHT = 0.8000  # Weight of first in combination of two XGB models

BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
import datetime as dt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.noise import GaussianDropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer



##### READ IN RAW DATA

print( "\nReading data from disk ...")
prop = pd.read_csv('../input/properties_2016.csv')
train = pd.read_csv("../input/train_2016_v2.csv")




################
################
##  LightGBM  ##
################
################

# This section is (I think) originally derived from SIDHARTH's script:
#   https://www.kaggle.com/sidharthkumar/trying-lightgbm
# which was forked and tuned by Yuqing Xue:
#   https://www.kaggle.com/yuqingxue/lightgbm-85-97
# and updated by me (Andy Harless):
#   https://www.kaggle.com/aharless/lightgbm-with-outliers-remaining
# and a lot of additional changes have happened since then

##### PROCESS DATA FOR LIGHTGBM

print( "\nProcessing data for LightGBM ..." )
for c, dtype in zip(prop.columns, prop.dtypes):	
    if dtype == np.float64:		
        prop[c] = prop[c].astype(np.float32)

df_train = train.merge(prop, how='left', on='parcelid')
df_train.fillna(df_train.median(),inplace = True)

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 
                         'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag'], axis=1)
#x_train['Ratio_1'] = x_train['taxvaluedollarcnt']/x_train['taxamount']
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
params['sub_feature'] = 0.345    # feature_fraction (small values => use very different submodels)
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params['verbose'] = 0
params['feature_fraction_seed'] = 2
params['bagging_seed'] = 3

np.random.seed(0)
random.seed(0)

print("\nFitting LightGBM model ...")
clf = lgb.train(params, d_train, 430)

del d_train; gc.collect()
del x_train; gc.collect()

print("\nPrepare for LightGBM prediction ...")
print("   Read sample file ...")
sample = pd.read_csv('../input/sample_submission.csv')
print("   ...")
sample['parcelid'] = sample['ParcelId']
print("   Merge with property data ...")
df_test = sample.merge(prop, on='parcelid', how='left')
print("   ...")
del sample, prop; gc.collect()
print("   ...")
#df_test['Ratio_1'] = df_test['taxvaluedollarcnt']/df_test['taxamount']
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




################
################
##  XGBoost   ##
################
################

# This section is (I think) originally derived from Infinite Wing's script:
#   https://www.kaggle.com/infinitewing/xgboost-without-outliers-lb-0-06463
# inspired by this thread:
#   https://www.kaggle.com/c/zillow-prize-1/discussion/33710
# but the code has gone through a lot of changes since then


##### RE-READ PROPERTIES FILE
##### (I tried keeping a copy, but the program crashed.)

print( "\nRe-reading properties file ...")
properties = pd.read_csv('../input/properties_2016.csv')



##### PROCESS DATA FOR XGBOOST

print( "\nProcessing data for XGBoost ...")
for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

train_df = train.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
x_test = properties.drop(['parcelid'], axis=1)
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
print("num_boost_rounds="+str(num_boost_rounds))

# train model
print( "\nTraining XGBoost ...")
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

print( "\nPredicting with XGBoost ...")
xgb_pred1 = model.predict(dtest)

print( "\nFirst XGBoost predictions:" )
print( pd.DataFrame(xgb_pred1).head() )



##### RUN XGBOOST AGAIN

print("\nSetting up data for XGBoost ...")
# xgboost params
xgb_params = {
    'eta': 0.033,
    'max_depth': 6,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
    'silent': 1
}

num_boost_rounds = 150
print("num_boost_rounds="+str(num_boost_rounds))

print( "\nTraining XGBoost again ...")
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

print( "\nPredicting with XGBoost again ...")
xgb_pred2 = model.predict(dtest)

print( "\nSecond XGBoost predictions:" )
print( pd.DataFrame(xgb_pred2).head() )



##### COMBINE XGBOOST RESULTS
xgb_pred = XGB1_WEIGHT*xgb_pred1 + (1-XGB1_WEIGHT)*xgb_pred2
#xgb_pred = xgb_pred1

print( "\nCombined XGBoost predictions:" )
print( pd.DataFrame(xgb_pred).head() )

del train_df
del x_train
del x_test
del properties
del dtest
del dtrain
del xgb_pred1
del xgb_pred2 
gc.collect()




######################
######################
##  Neural Network  ##
######################
######################

# Neural network copied from this script:
#   https://www.kaggle.com/aharless/keras-neural-network-lb-06492 (version 20)
# which was built on the skeleton in this notebook:
#   https://www.kaggle.com/prasunmishra/ann-using-keras


# Read in data for neural network
print( "\n\nProcessing data for Neural Network ...")
print('\nLoading train, prop and sample data...')
train = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])
prop = pd.read_csv('../input/properties_2016.csv')
sample = pd.read_csv('../input/sample_submission.csv')
 
print('Fitting Label Encoder on properties...')
for c in prop.columns:
    prop[c]=prop[c].fillna(-1)
    if prop[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(prop[c].values))
        prop[c] = lbl.transform(list(prop[c].values))
        
print('Creating training set...')
df_train = train.merge(prop, how='left', on='parcelid')

df_train["transactiondate"] = pd.to_datetime(df_train["transactiondate"])
df_train["transactiondate_year"] = df_train["transactiondate"].dt.year
df_train["transactiondate_month"] = df_train["transactiondate"].dt.month
df_train['transactiondate_quarter'] = df_train['transactiondate'].dt.quarter
df_train["transactiondate"] = df_train["transactiondate"].dt.day

print('Filling NA/NaN values...' )
df_train.fillna(-1.0)

print('Creating x_train and y_train from df_train...' )
x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode','fireplacecnt', 'fireplaceflag'], axis=1)
y_train = df_train["logerror"]

y_mean = np.mean(y_train)
print(x_train.shape, y_train.shape)
train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

print('Creating df_test...')
sample['parcelid'] = sample['ParcelId']

print("Merging Sample with property data...")
df_test = sample.merge(prop, on='parcelid', how='left')

df_test["transactiondate"] = pd.to_datetime('2016-11-15')  # placeholder value for preliminary version
df_test["transactiondate_year"] = df_test["transactiondate"].dt.year
df_test["transactiondate_month"] = df_test["transactiondate"].dt.month
df_test['transactiondate_quarter'] = df_test['transactiondate'].dt.quarter
df_test["transactiondate"] = df_test["transactiondate"].dt.day     
x_test = df_test[train_columns]

print('Shape of x_test:', x_test.shape)
print("Preparing x_test...")
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
  

## Preprocessing
print("\nPreprocessing neural network data...")
imputer= Imputer()
imputer.fit(x_train.iloc[:, :])
x_train = imputer.transform(x_train.iloc[:, :])
imputer.fit(x_test.iloc[:, :])
x_test = imputer.transform(x_test.iloc[:, :])

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

len_x=int(x_train.shape[1])
print("len_x is:",len_x)


# Neural Network
print("\nSetting up neural network model...")
nn = Sequential()
nn.add(Dense(units = 400 , kernel_initializer = 'normal', input_dim = len_x))
nn.add(PReLU())
nn.add(Dropout(.4))
nn.add(Dense(units = 160 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.6))
nn.add(Dense(units = 64 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.5))
nn.add(Dense(units = 26, kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.6))
nn.add(Dense(1, kernel_initializer='normal'))
nn.compile(loss='mae', optimizer=Adam(lr=4e-3, decay=1e-4))

print("\nFitting neural network model...")
nn.fit(np.array(x_train), np.array(y_train), batch_size = 32, epochs = 70, verbose=2)

print("\nPredicting with neural network model...")
#print("x_test.shape:",x_test.shape)
y_pred_ann = nn.predict(x_test)

print( "\nPreparing results for write..." )
nn_pred = y_pred_ann.flatten()
print( "Type of nn_pred is ", type(nn_pred) )
print( "Shape of nn_pred is ", nn_pred.shape )

print( "\nNeural Network predictions:" )
print( pd.DataFrame(nn_pred).head() )


# Cleanup
del train
del prop
del sample
del x_train
del x_test
del df_train
del df_test
del y_pred_ann
gc.collect()




################
################
##    OLS     ##
################
################

# This section is derived from the1owl's notebook:
#    https://www.kaggle.com/the1owl/primer-for-the-zillow-pred-approach
# which I (Andy Harless) updated and made into a script:
#    https://www.kaggle.com/aharless/updated-script-version-of-the1owl-s-basic-ols

np.random.seed(17)
random.seed(17)

print( "\n\nProcessing data for OLS ...")

train = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])
properties = pd.read_csv("../input/properties_2016.csv")
submission = pd.read_csv("../input/sample_submission.csv")
print(len(train),len(properties),len(submission))

def get_features(df):
    df["transactiondate"] = pd.to_datetime(df["transactiondate"])
    df["transactiondate_year"] = df["transactiondate"].dt.year
    df["transactiondate_month"] = df["transactiondate"].dt.month
    df['transactiondate'] = df['transactiondate'].dt.quarter
    df = df.fillna(-1.0)
    return df

def MAE(y, ypred):
    #logerror=log(Zestimate)−log(SalePrice)
    return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)

train = pd.merge(train, properties, how='left', on='parcelid')
y = train['logerror'].values
test = pd.merge(submission, properties, how='left', left_on='ParcelId', right_on='parcelid')
properties = [] #memory

exc = [train.columns[c] for c in range(len(train.columns)) if train.dtypes[c] == 'O'] + ['logerror','parcelid']
col = [c for c in train.columns if c not in exc]

train = get_features(train[col])
test['transactiondate'] = '2016-01-01' #should use the most common training date
test = get_features(test[col])


print("\nFitting OLS...")
reg = LinearRegression(n_jobs=-1)
reg.fit(train, y); print('fit...')
print(MAE(y, reg.predict(train)))
train = [];  y = [] #memory

test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
test_columns = ['201610','201611','201612','201710','201711','201712']




########################
########################
##  Combine and Save  ##
########################
########################


##### COMBINE PREDICTIONS

print( "\nCombining XGBoost, LightGBM, NN, and baseline predicitons ..." )
lgb_weight = 1 - XGB_WEIGHT - BASELINE_WEIGHT - NN_WEIGHT - OLS_WEIGHT 
lgb_weight0 = lgb_weight / (1 - OLS_WEIGHT)
xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
baseline_weight0 =  BASELINE_WEIGHT / (1 - OLS_WEIGHT)
nn_weight0 = NN_WEIGHT / (1 - OLS_WEIGHT)
pred0 = 0
pred0 += xgb_weight0*xgb_pred
pred0 += baseline_weight0*BASELINE_PRED
pred0 += lgb_weight0*p_test
pred0 += nn_weight0*nn_pred

print( "\nCombined XGB/LGB/NN/baseline predictions:" )
print( pd.DataFrame(pred0).head() )

print( "\nPredicting with OLS and combining with XGB/LGB/NN/baseline predicitons: ..." )
for i in range(len(test_dates)):
    test['transactiondate'] = test_dates[i]
    pred = FUDGE_FACTOR * ( OLS_WEIGHT*reg.predict(get_features(test)) + (1-OLS_WEIGHT)*pred0 )
    submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
    print('predict...', i)

print( "\nCombined XGB/LGB/NN/baseline/OLS predictions:" )
print( submission.head() )



##### WRITE THE RESULTS

from datetime import datetime

print( "\nWriting results to disk ..." )
submission.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print( "\nFinished ...")




########################
########################
##  Version Comments  ##
########################
########################

# version 1: forked from "XGBoost, LightGBM, and OLS"
# version 2: deleted old comments and added keras imports
# version 3: added NN code (not using result yet)
# version 5: use results of NN with 0.1 weight (arbitrary first guess)
# version 7: reduce neural network weight to 0.05 and adjust others proportionally
# version 8: reduce NN weight to 0.03
# version 9: reduce NN weight to 0.01
# version 10: increasee NN weight to 0.02
# version 11: change NN parameters, increase weight to .03, and reduce OLS weight to .07
# version 12: change NN parameters again & adjust various weights
# version 13: more weight adjustments (NN back down to .02)
# version 15: PReLU activations, learning rate decay
# version 16: OLS weight up, XGB weight down, baseline weight up, increase dropout
# version 17: weights change in opposite direction
# version 18: change weights further in that direction
# version 20: revert to version 17 weights & fix test dates assignment error
# version 23: Fix date assignment error, hopefully correctly this time
# version 32: Reduce NN weight to .015
# version 33: Increase NN weight to .025
# version 34: Change weighting scheme to fix error (need to subtract OLS_WEIGHT from XGB_WEIGHT)
# version 35: Increase NN weight .03 to .04, LB .0643988 -> .0643965
# versein 36: Try NN weight .05 -> LB .0643917
# version 37: Try NN weight .07 -> LB .0643920 
# version 38: NN_WEIGHT=.06, XGB_WEIGHT=.58 -> LB .0643949
# version 39: XGB_WEIGHT=.62 -> .LB 0632901
# version 40: BASELINE_WEIGHT 0.01 to 0 -> LB .0643908
# version 41: BASELINE_WEIGHT=.02 -> LB .0643924
# version 42: OLS_WEIGHT .07 to .08 -> LB .0643925
# version 43: BASELINE_WEIGHT=.01, OLS_WEIGHT=.075 -> LB .0643919
# version 44: OLS_WEIGHT=.06 -> LB .0643926
# version 45: OLS_WEIGHT=.07, impute NAs in LGB test set -> LB .06440
# version 46: get rid of imputation, change to Gaussian dropout -> lB .0643909
# version 47: Add fudge factor, back to regular dropout -> LB .0643910
# version 48: Decrease fudge factor 1.004 to 1.002 -> LB .0643938
# version 49: Increase fudge factor to 1.01 -> LB .0643917
# version 50: Try FUDGE_FACTOR=.99 (equivalent to using zero baseline in ensebmle) -> LB .063974
# version 51: FUDGE_FACTOR=1.0, should be equivalent to version 39, except for randomness
# version 52: FUDGB_FACTOR=1.03, rough optimum based on Q4 CV for XGB alone
#             (http://www.kaggle.com/aharless/xgboost-using-q4-for-validation-original-retuned/)
# version 54: Try FUDGE_FACTOR=1.06
# version 55: Try FUDGE_FACTOR=1.12
# version 56: Try FUDGE_FACTOR=1.15
# version 57: FUDGE_FACTOR back to 1.12, try NN_WEIGHT=0.07
# version 58: Increase NN_WEIGHT to 0.08
# version 59: Reduce OLS_WEIGHT to 0.06
# version 60: Reduce OLS_WEIGHT to 0.05
# version 61: OLS_WEIGHT=.062