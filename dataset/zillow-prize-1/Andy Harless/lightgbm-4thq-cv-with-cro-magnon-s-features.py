######

## CV bfor 1 LigthGBM
### Train set: Month 1 to 9
### Val set: Month 10 to 12


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
params['learning_rate'] = 0.0015 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'          # or 'mae'
params['sub_feature'] = 0.3      # feature_fraction (small values => use very different submodels)
params['bagging_fraction'] = 0.8 # sub_row
params['bagging_freq'] = 50
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 320         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params['verbose'] = 0
params['feature_fraction_seed'] = 2
params['bagging_seed'] = 3

np.random.seed(0)
random.seed(0)

print("\nFitting LightGBM model ...")

clf = lgb.train(params, d_train, 1000)


print("   ...")


df_test = xval

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


# version 7:  learning_rate .0021->.0015, rounds 430->1000
# version 8:  bagging_fraction .85->.80
# version 9:  bagging_freq 40->50
# version 10:  min_data 500->350
# version 11:  min_data 350->320