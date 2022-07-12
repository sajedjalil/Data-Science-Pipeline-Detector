# import libraries
import numpy as np
import pandas as pd
import os
import time
import lightgbm as lgb
from sklearn.model_selection import KFold
import matplotlib.mlab as mlab

def add_agg(merged_df,gr_cols,new_col_name,incr_yr):
    agg1 = train_df.groupby(gr_cols)['target'].agg('mean').reset_index()
    cols2 = gr_cols.copy()
    cols2.append(new_col_name)
    agg1.columns = cols2
    if incr_yr:
        agg1['year']+=1
    merged_df = pd.merge(merged_df,agg1,how='left',left_on=gr_cols,right_on=gr_cols)
    return merged_df

# read raw data
start_time = time.time()
input_dir = os.path.join(os.pardir, 'input')
train_df = pd.read_csv(os.path.join(input_dir,'train.csv'), nrows=None)
test_df = pd.read_csv(os.path.join(input_dir,'test.csv'), nrows=None)

# add columns: mean and median related
#train_df['store_item_mean'] = train_df.groupby(['store','item'])['sales'].transform('mean').astype(np.float16)
#train_df['store_item_median'] = train_df.groupby(['store','item'])['sales'].transform('median').astype(np.float16)

# add columns
#train_df['lag_t7']            = train_df.groupby(['item','store'])['sales'].transform(lambda x: x.shift(7))
#train_df['lag_t14']           = train_df.groupby(['item','store'])['sales'].transform(lambda x: x.shift(14))
#train_df['lag_t28']           = train_df.groupby(['item','store'])['sales'].transform(lambda x: x.shift(28))
#train_df['rolling_mean_t7']   = train_df.groupby(['item','store'])['sales'].transform(lambda x: x.shift(28).rolling(7).mean())
#train_df['rolling_std_t7']    = train_df.groupby(['item','store'])['sales'].transform(lambda x: x.shift(28).rolling(7).std())
#train_df['rolling_mean_t30']  = train_df.groupby(['item','store'])['sales'].transform(lambda x: x.shift(28).rolling(30).mean())
#train_df['rolling_std_t30']   = train_df.groupby(['item','store'])['sales'].transform(lambda x: x.shift(28).rolling(30).std())
#train_df['rolling_mean_t90']  = train_df.groupby(['item','store'])['sales'].transform(lambda x: x.shift(28).rolling(90).mean())
#train_df['rolling_std_t90']   = train_df.groupby(['item','store'])['sales'].transform(lambda x: x.shift(28).rolling(90).std())
#train_df['rolling_mean_t180'] = train_df.groupby(['item','store'])['sales'].transform(lambda x: x.shift(28).rolling(180).mean())
#train_df['rolling_std_t180']  = train_df.groupby(['item','store'])['sales'].transform(lambda x: x.shift(28).rolling(180).std())
#train_df['rolling_skew_t30']  = train_df.groupby(['item','store'])['sales'].transform(lambda x: x.shift(28).rolling(30).skew())
#train_df['rolling_kurt_t30']  = train_df.groupby(['item','store'])['sales'].transform(lambda x: x.shift(28).rolling(30).kurt())

# Merge test/train datasets into a single one and separate unneeded columns
merged_df = pd.concat([train_df,test_df],sort=False)

# add columns: date related
merged_df['date'] = pd.to_datetime(merged_df['date'],infer_datetime_format=True)
merged_df['year'] = merged_df['date'].dt.year
merged_df['month'] = merged_df['date'].dt.month
merged_df['day'] = merged_df['date'].dt.dayofweek
merged_df['quarter'] = merged_df['date'].dt.quarter
#merged_df['dayofweek'] = merged_df['date'].dt.dayofweek
merged_df['weekofyear'] = merged_df['date'].dt.weekofyear
#merged_df['dayofyear'] = merged_df['date'].dt.dayofyear
merged_df.drop('date', axis=1, inplace=True)

# add grouped columns
train_df=pd.DataFrame(merged_df[merged_df.sales.notna()].values)
train_df.columns=merged_df.columns
train_df['target']=train_df['sales'] 

# scale for item+store+year(prev). Need to scale predictions back up
merged_df = add_agg(merged_df,['item','store','year'],'tsy',1)
merged_df['sales']/=merged_df['tsy']
merged_df = merged_df[merged_df.year>2013]
tsy=merged_df.pop('tsy')

# pop sales and ID
ID=merged_df[merged_df.id.notna()]['id']
target=merged_df[merged_df.sales.notna()]['sales']
merged_df.drop(['id','sales'], axis=1, inplace=True)
len_train=target.shape[0]

# specify your configurations as a dict
params = {
    'nthread': 10,
    'max_depth': 6,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.25,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.06,
    'lambda_l2': 0.1,
    'verbose': -1
}

# training
num_folds = 5
test_x = merged_df[len_train:].values
all_x = merged_df[:len_train].values
all_y = target.values
oof_preds = np.zeros([all_y.shape[0]])
sub_preds = np.zeros([test_x.shape[0]])
folds = KFold(n_splits=num_folds, shuffle=True, random_state=345665)
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(all_x)):
    train_x, train_y = all_x[train_idx], all_y[train_idx]
    valid_x, valid_y = all_x[valid_idx], all_y[valid_idx]
    lgb_train = lgb.Dataset(train_x,train_y)
    lgb_valid = lgb.Dataset(valid_x,valid_y)

    gbm = lgb.train(params, lgb_train, 1000, 
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=100, verbose_eval=100)
    oof_preds[valid_idx] = gbm.predict(valid_x, num_iteration=gbm.best_iteration)
    sub_preds[:] += gbm.predict(test_x, num_iteration=gbm.best_iteration) / folds.n_splits
    valid_idx += 1

# Write submission file
pred = (sub_preds * tsy[len_train:] ).astype(np.float32)
out_df = pd.DataFrame({'id': ID.astype(np.int32), 'sales': pred})
out_df.to_csv('submission.csv', index=False)