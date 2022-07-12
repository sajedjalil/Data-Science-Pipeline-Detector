#All import statements
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

#Load All the Datasets
print("Loading Datasets")
train = pd.read_csv('../input/train_2016_v2.csv')
properties = pd.read_csv('../input/properties_2016.csv',low_memory=False)
sample = pd.read_csv('../input/sample_submission.csv')
print("Datasets Loaded")

#Make own feature set
print('Making own feature set')
for c, dtype in zip(properties.columns, properties.dtypes):
    if dtype == np.float64:
        properties[c] = properties[c].astype(np.float32)

for c in properties.dtypes[properties.dtypes == object].index.values:
    properties[c] = (properties[c] == True)
        
properties['livingArea'] = properties['calculatedfinishedsquarefeet'] / properties[
        'finishedsquarefeet12']

ulimit = np.percentile(train.logerror.values, 99)
llimit = np.percentile(train.logerror.values, 1)
train[ train.logerror < llimit ] = llimit
train[ train.logerror > ulimit ] = ulimit

col = "taxamount"
ulimit = np.percentile(properties[col].values, 99.5)
llimit = np.percentile(properties[col].values, 0.5)
properties[col].loc[properties[col]>ulimit] = ulimit
properties[col].loc[properties[col]<llimit] = llimit

properties['bedroomcnt'].loc[properties['bedroomcnt']>7] = 7

col = "calculatedfinishedsquarefeet"
ulimit = np.percentile(properties[col].values, 99.5)
llimit = np.percentile(properties[col].values, 0.5)
properties[col].loc[properties[col]>ulimit] = ulimit
properties[col].loc[properties[col]<llimit] = llimit

col = "finishedsquarefeet12"
ulimit = np.percentile(properties[col].values, 99.5)
llimit = np.percentile(properties[col].values, 0.5)
properties[col].loc[properties[col]>ulimit] = ulimit
properties[col].loc[properties[col]<llimit] = llimit
        
df_train = train.merge(properties, how='left', on='parcelid')
df_train = df_train.drop(['parcelid', 'transactiondate','logerror'], axis=1)

dropcols = ['finishedsquarefeet12','finishedsquarefeet13', 'finishedsquarefeet15',
            'finishedsquarefeet6']
dropcols.append('finishedsquarefeet50')
dropcols.append('calculatedbathnbr')
dropcols.append('fullbathcnt')
index = df_train.hashottuborspa.isnull()
df_train.loc[index,'hashottuborspa'] = np.nan
dropcols.append('pooltypeid10')
index = df_train.pooltypeid2.isnull()
df_train.loc[index,'pooltypeid2'] = 0
index = df_train.pooltypeid7.isnull()
df_train.loc[index,'pooltypeid7'] = 0
index = df_train.poolcnt.isnull()
df_train.loc[index,'poolcnt'] = 0
poolsizesum_median = df_train.loc[df_train['poolcnt'] > 0, 'poolsizesum'].median()
df_train.loc[(df_train['poolcnt'] > 0) & (df_train['poolsizesum'].isnull()), 'poolsizesum'] = poolsizesum_median
df_train['fireplaceflag']= False
df_train.loc[df_train['fireplacecnt']>0,'fireplaceflag']= True
index = df_train.fireplacecnt.isnull()
df_train.loc[index,'fireplacecnt'] = 0
index = df_train.taxdelinquencyflag.isnull()
df_train.loc[index,'taxdelinquencyflag'] = np.nan
index = df_train.garagecarcnt.isnull()
df_train.loc[index,'garagecarcnt'] = 0
index = df_train.garagetotalsqft.isnull()
df_train.loc[index,'garagetotalsqft'] = 0
df_train['airconditioningtypeid'].value_counts()
index = df_train.airconditioningtypeid.isnull()
df_train.loc[index,'airconditioningtypeid'] = 1
index = df_train.heatingorsystemtypeid.isnull()
df_train.loc[index,'heatingorsystemtypeid'] = 2
index = df_train.threequarterbathnbr.isnull()
df_train.loc[index,'threequarterbathnbr'] = 1

missingvalues_prop = (df_train.isnull().sum()/len(df_train)).reset_index()
missingvalues_prop.columns = ['field','proportion']
missingvalues_prop = missingvalues_prop.sort_values(by = 'proportion', ascending = False)
missingvaluescols = missingvalues_prop[missingvalues_prop['proportion'] > 0.97].field.tolist()
dropcols = dropcols + missingvaluescols
X_train = df_train.drop(dropcols, axis=1)
train_cols = list(X_train.columns)
print("Done")

Y_train = np.array(train["logerror"])
y_mean = np.mean(Y_train)
sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(properties, on='parcelid', how='left')
X_test = df_test[train_cols]
print(X_train.shape, Y_train.shape, X_test.shape)
print("Finished Making features")

#Clean Data
print('cleaning Data')
imputer= Imputer(strategy='most_frequent')
imputer.fit(X_train.iloc[:, :])
X_train = imputer.transform(X_train.iloc[:,:])
print("X_train done")
for c in X_test.columns:
        #X_test[c]=X_test[c].fillna(-1)
        X_test[c].fillna(X_test[c].mode())
        if X_test[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(X_test[c].values))
            X_test[c] = lbl.transform(list(X_test[c].values))
print('filled missing values')
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


xgb_params = {
    'eta': 0.007,
    'max_depth': 7,
    'subsample': 0.60,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 5.0,
    'alpha': 0.65,
    'colsample_bytree': 0.5,
    'base_score': y_mean,
    'silent': 1
}
dtrain = xgb.DMatrix(X_train, Y_train)
dtest = xgb.DMatrix(X_test)

early_stopping_rounds = round( 1 / xgb_params['eta'] )
num_boost_rounds = round( 20 / xgb_params['eta'] )
# cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   nfold=5,
                   num_boost_round=num_boost_rounds,
                   early_stopping_rounds=early_stopping_rounds,
                   verbose_eval=10, 
                   show_stdv=False
                  )
num_boost_rounds = len(cv_result)
print(num_boost_rounds)
# train model
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
pred = model.predict(dtest)
y_pred=[]

for i,predict in enumerate(pred):
    y_pred.append(str(round(predict,4)))
y_pred=np.array(y_pred)

#Writing data to csv
print("Writing data to csv file")
output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
output.to_csv('Using_XGBoost_Regressor{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
print( "\nFinished!" )


