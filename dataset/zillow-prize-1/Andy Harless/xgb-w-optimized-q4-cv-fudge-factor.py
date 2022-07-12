FUDGE_FACTOR = 1.033

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import datetime as dt
import gc

properties = pd.read_csv('../input/properties_2016.csv')
train = pd.read_csv("../input/train_2016_v2.csv")
for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

train_df = train.merge(properties, how='left', on='parcelid')
select_qtr4 = pd.to_datetime(train_df["transactiondate"]).dt.month > 9
x_train_all = train_df.drop(['parcelid', 'logerror','transactiondate',
                             'airconditioningtypeid', 'buildingclasstypeid',
                             'buildingqualitytypeid', 'regionidcity'], axis=1)
x_valid = x_train_all[select_qtr4]
x_train = x_train_all[~select_qtr4]
y_valid = train_df["logerror"].values.astype(np.float32)[select_qtr4]
x_test = properties.drop(['parcelid','airconditioningtypeid', 'buildingclasstypeid',
                          'buildingqualitytypeid', 'regionidcity'], axis=1)
print('Shape full train: {}'.format(x_train_all.shape))
print('Shape train: {}\nShape valid: {}'.format(x_train.shape, x_valid.shape))
print('Shape valid y: {}'.format(y_valid.shape))
print('Shape test: {}'.format(x_test.shape))

del train
del x_train_all
gc.collect()

train_df=train_df[~select_qtr4]
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.419 ]
x_train=train_df.drop(['parcelid', 'logerror','transactiondate',
                       'airconditioningtypeid', 'buildingclasstypeid',
                       'buildingqualitytypeid', 'regionidcity'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)
print('After removing outliers:')     
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
print('Shape train y: {}'.format(y_train.shape))

del train_df
del select_qtr4
gc.collect()

xgb_params = {  # Baseline 64853 parameters
    'eta': 0.007,
    'max_depth': 6, 
    'subsample': 0.5,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 8.0,
    'alpha': 0.8,
    'colsample_bytree': 0.7,
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dvalid_x = xgb.DMatrix(x_valid)
dvalid_xy = xgb.DMatrix(x_valid, y_valid)
dtest = xgb.DMatrix(x_test)

evals = [(dtrain,'train'),(dvalid_xy,'eval')]
num_boost_rounds=4000
model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds,
                  evals=evals, early_stopping_rounds=150, verbose_eval=10)
                  
valid_pred = model.predict(dvalid_x, ntree_limit=model.best_ntree_limit)
print( "XGBoost validation set predictions:" )
print( pd.DataFrame(valid_pred).head() )

print( "\nMean absolute validation error, with fudge factor: ")
print(mean_absolute_error(y_valid, FUDGE_FACTOR*valid_pred))

pred = FUDGE_FACTOR*model.predict(dtest, ntree_limit=model.best_ntree_limit)
print( "\nXGBoost test set predictions:" )
print( pd.DataFrame(pred).head() )

y_pred=[]

for i,predict in enumerate(pred):
    y_pred.append(str(round(predict,4)))
y_pred=np.array(y_pred)

output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})
# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
from datetime import datetime
output.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)