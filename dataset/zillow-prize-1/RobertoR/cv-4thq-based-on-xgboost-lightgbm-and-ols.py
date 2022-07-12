######

## CV bfor 1 LigthGBM and 2 XGB
### Train set: Month 1 to 9
### Val set: Month 10 to 12


# Parameters
#XGB_WEIGHT = 0.6700
#BASELINE_WEIGHT = 0.0056
#OLS_WEIGHT = 0.0550

XGB_WEIGHT = 0.6415
BASELINE_WEIGHT = 0.0050
OLS_WEIGHT = 0.0828

XGB1_WEIGHT = 0.8083  # Weight of first in combination of two XGB models

BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg


#XGB_WEIGHT = 0.6415
#BASELINE_WEIGHT = 0.0056
#OLS_WEIGHT = 0.0828

#XGB1_WEIGHT = 0.8083  # Weight of first in combination of two XGB models

#BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
import datetime as dt
from sklearn.metrics import mean_absolute_error

##### READ IN RAW DATA

print( "\nReading data from disk ...")
prop = pd.read_csv('../input/properties_2016.csv')
train = pd.read_csv("../input/train_2016_v2.csv")


print( "\nProcessing data for LightGBM ..." )


for c, dtype in zip(prop.columns, prop.dtypes):	
    if dtype == np.float64:		
        prop[c] = prop[c].astype(np.float32)


print("\nPreparing train set...")



#########################################################################################3
####Ofert1: LIGTH to 0.0652813
ofert1 = prop.groupby(['yearbuilt', 'bedroomcnt', 'regionidcity'],  as_index=False)['parcelid'].count()
ofert1=pd.DataFrame(ofert1)
ofert1.columns.values[3] = 'count_ParcelId'  
prop= pd.merge(prop,ofert1, on=['yearbuilt', 'bedroomcnt', 'regionidcity'], how='left')



####Ofert2: v12
ofert2 = prop.groupby(['yearbuilt', 'roomcnt', 'regionidcity'],  as_index=False)['parcelid'].count()
ofert2=pd.DataFrame(ofert2)
ofert2.columns.values[3] = 'count_ParcelId_Of2'  
prop= pd.merge(prop,ofert2, on=['yearbuilt', 'roomcnt', 'regionidcity'], how='left')






####Tax1: ####Ofert1: LIGTH to 0.0652813

Tax1 = prop.groupby(['yearbuilt', 'bedroomcnt', 'regionidcity'],  as_index=False)['taxamount'].mean()
Tax1=pd.DataFrame(Tax1)
Tax1.columns.values[3] = 'mean_TaxAmount'  
prop= pd.merge(prop,Tax1, on=['yearbuilt', 'bedroomcnt', 'regionidcity'], how='left')


#######################################################################################


print

df_train = train.merge(prop, how='left', on='parcelid')
df_train.fillna(df_train.median(),inplace = True)


df_train["transactiondate"] = pd.to_datetime(df_train["transactiondate"])
df_train["Month"] = df_train["transactiondate"].dt.month
 
x_trainT = df_train.drop(['parcelid', 'propertyzoningdesc', 
                         'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag',
                         'transactiondate'], axis=1)
                         
 
x_train = x_trainT[x_trainT["Month"]<10]                       
xval = x_trainT[x_trainT["Month"]>=10]  

            

y_train = x_train['logerror'].values
print(x_train.shape, y_train.shape)


x_train2 = x_train.drop("logerror", axis=1)
train_columns = x_train2.columns

for c in x_train2.dtypes[x_train2.dtypes == object].index.values:
    x_train2[c] = (x_train2[c] == True)

x_train2 = x_train2.values.astype(np.float32, copy=False)

d_train = lgb.Dataset(x_train2, label=y_train)



print("\nRunning LIGHTGBM.......")

##### RUN LIGHTGBM

params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'          # or 'mae'
params['sub_feature'] = 0.3      # feature_fraction (small values => use very different submodels)
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


print("   ...")


df_test = xval

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
logReal=xval.logerror
del x_test; gc.collect()

print( "\nUnadjusted LightGBM predictions:" )
print( pd.DataFrame(p_test).head() )


maelgh=mean_absolute_error(logReal, p_test)


print("##########")
print("CV: MAE for LIGHTGBM is", maelgh)
print("##########")

################
################
##  XGBoost   ##
################
################


del prop

print( "\nRe-reading properties file ...")
properties =pd.read_csv('../input/properties_2016.csv')




#########################################################################################3
####Ofert1: LIGTH to 0.0652813
#ofert1 = properties.groupby(['yearbuilt', 'bedroomcnt', 'regionidcity'],  as_index=False)['parcelid'].count()
#ofert1=pd.DataFrame(ofert1)
#ofert1.columns.values[3] = 'count_ParcelId'  
#properties= pd.merge(properties,ofert1, on=['yearbuilt', 'bedroomcnt', 'regionidcity'], how='left')



####Ofert2: v12
#ofert2 = properties.groupby(['yearbuilt', 'roomcnt', 'regionidcity'],  as_index=False)['parcelid'].count()
#ofert2=pd.DataFrame(ofert2)
#ofert2.columns.values[3] = 'count_ParcelId_Of2'  
#properties= pd.merge(properties,ofert2, on=['yearbuilt', 'roomcnt', 'regionidcity'], how='left')


####Ofert3:v12
#ofert3 = properties.groupby(['yearbuilt', 'bathroomcnt', 'regionidcity'],  as_index=False)['parcelid'].count()
#ofert3=pd.DataFrame(ofert3)
#ofert3.columns.values[3] = 'count_ParcelId_Of3'  
#properties= pd.merge(properties,ofert3, on=['yearbuilt', 'bathroomcnt', 'regionidcity'], how='left')


####Ofert4: v12
#ofert4 = properties.groupby(['yearbuilt', 'finishedsquarefeet12', 'regionidcity'],  as_index=False)['parcelid'].count()
#ofert4=pd.DataFrame(ofert4)
#ofert4.columns.values[3] = 'count_ParcelId_Of4'  
#properties= pd.merge(properties,ofert4, on=['yearbuilt', 'finishedsquarefeet12', 'regionidcity'], how='left')




####Tax1: ####Ofert1: LIGTH to 0.0652813

#Tax1 = properties.groupby(['yearbuilt', 'bedroomcnt', 'regionidcity'],  as_index=False)['taxamount'].mean()
#Tax1=pd.DataFrame(Tax1)
#Tax1.columns.values[3] = 'mean_TaxAmount'  
#properties= pd.merge(properties,Tax1, on=['yearbuilt', 'bedroomcnt', 'regionidcity'], how='left')


#######################################################################################



print( "\nProcessing data for XGBoost ...")
for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))



train_df = train.merge(properties, how='left', on='parcelid')


train_df["transactiondate"] = pd.to_datetime(train_df["transactiondate"])
train_df["Month"] = train_df["transactiondate"].dt.month

x_trainT = train_df.drop(['parcelid', 'propertyzoningdesc', 
                         'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag','transactiondate'], axis=1)
                         
 
#Subset train y validation set
 
x_train = x_trainT[x_trainT["Month"]<10]                       
x_val   = x_trainT[x_trainT["Month"]>=10]  

y_train = x_train['logerror'].values
print(x_train.shape, y_train.shape)

len(y_train)
len(x_train)

# shape        
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_val.shape))


# drop out ouliers
x_train=x_train[x_train.logerror > -0.4 ]
x_train=x_train[x_train.logerror < 0.419 ]

#####

x_val2 = x_val.drop("logerror", axis=1)

x_train2=x_train.drop(['logerror'], axis=1)

train_columns = x_train2.columns


y_train = x_train['logerror'].values


y_mean = np.mean(y_train)

print('After removing outliers:')     
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_val.shape))



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


len(x_train2)
dtrain = xgb.DMatrix(x_train2, y_train)
dtest = xgb.DMatrix(x_val2)


num_boost_rounds = 500
print("num_boost_rounds="+str(num_boost_rounds))



# train model
print( "\nTraining XGBoost ...")
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

print( "\nPredicting with XGBoost ...")
xgb_pred1 = model.predict(dtest)

print( "\nFirst XGBoost predictions:" )
print( pd.DataFrame(xgb_pred1).head() )



maexgb=mean_absolute_error(x_val.logerror,xgb_pred1)



print("##########")
print("CV: MAE FIRST XGB is:", maexgb)
print("##########")

#### RUN XGBOOST AGAIN

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

num_boost_rounds = 500
print("num_boost_rounds="+str(num_boost_rounds))

print( "\nTraining XGBoost again ...")
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

print( "\nPredicting with XGBoost again ...")
xgb_pred2 = model.predict(dtest)

print( "\nSecond XGBoost predictions:" )
print( pd.DataFrame(xgb_pred2).head() )


maexgb2=mean_absolute_error(x_val.logerror,xgb_pred2)



print("##########")
print("CV: MAE SECOND XGB is:", maexgb2)
print("##########")



##### COMBINE XGBOOST RESULTS
xgb_pred = XGB1_WEIGHT*xgb_pred1 + (1-XGB1_WEIGHT)*xgb_pred2
#xgb_pred = xgb_pred1

print( "\nCombined XGBoost predictions:" )
print( pd.DataFrame(xgb_pred).head() )



maexgbC=mean_absolute_error(x_val.logerror,xgb_pred)

print("CV es", maexgbC)


print( "\nCombining XGBoost, LightGBM, and baseline predicitons ..." )
lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT) / (1 - OLS_WEIGHT)
xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
baseline_weight0 =  BASELINE_WEIGHT / (1 - OLS_WEIGHT)

pred0 = xgb_weight0*xgb_pred + baseline_weight0*BASELINE_PRED + lgb_weight*p_test

maexgb3=mean_absolute_error(x_val.logerror,pred0)


print("##########")

print("CV: FINAL MAE is:", maexgb3)

print("##########")

#####
#Version 1: 0.064840

######
#Version10: 0.0648305 with Feat Eng. 
#Tax1+Ofert1---->. Transport info to Or kernel. Make submission.
######
##V17 : CV: FINAL MAE is: 0.0648308712264

######

#V21LB0.0643556 

###v222

###