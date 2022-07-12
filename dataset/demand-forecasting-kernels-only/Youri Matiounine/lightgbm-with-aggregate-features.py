# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import time
import lightgbm as lgb
from sklearn.model_selection import KFold

# add aggregate column to the data
def add_agg(merged_df, gr_cols, new_col_name, incr_yr):
    agg1 = train_df.groupby(gr_cols)['target'].agg('mean').reset_index()
    cols2 = gr_cols.copy()
    cols2.append(new_col_name)
    agg1.columns = cols2
    if incr_yr:
        agg1['year']+=1
    merged_df = pd.merge(merged_df, agg1, how='left', left_on=gr_cols, right_on=gr_cols)
    return merged_df


# read raw data
start_time = time.time()
print('  Loading data...')
input_dir  = os.path.join(os.pardir, 'input')
train_df   = pd.read_csv(os.path.join(input_dir, 'train.csv'), nrows=None)
test_df    = pd.read_csv(os.path.join(input_dir, 'test.csv'), nrows=None)
print('    Time elapsed %.0f sec'%(time.time()-start_time))

# Merge test/train datasets into a single one and separate unneeded columns
merged_df = pd.concat([train_df, test_df], sort=False)

# add columns: date related
merged_df['date']  = pd.to_datetime(merged_df['date'],infer_datetime_format=True)
merged_df['year']  = merged_df['date'].dt.year
merged_df['month']  = merged_df['date'].dt.month
merged_df['day']   = merged_df['date'].dt.dayofweek
merged_df.drop('date', axis=1, inplace=True)


# add grouped columns
train_df = pd.DataFrame(merged_df[merged_df.sales.notna()].values)
train_df.columns = merged_df.columns
train_df['target'] = train_df['sales'] # rename





# scale for item+store+month+year(prev). Need to scale predictions back up!
merged_df = add_agg(merged_df,['item','store','year'], 'tsy', 1)
merged_df['sales'] /= merged_df['tsy']
merged_df = add_agg(merged_df,['day','month','year'], 'dmy', 1)
merged_df['sales'] /= merged_df['dmy']
merged_df = merged_df[merged_df.year>2013]
tsy = merged_df.pop('tsy') * merged_df.pop('dmy')





# pop sales and ID
ID = merged_df[merged_df.id.notna()]['id']
target = merged_df[merged_df.sales.notna()]['sales']
merged_df.drop(['id','sales'], axis=1, inplace=True)
#merged_df.drop(['year','day','store','item'], axis=1, inplace=True)
len_train = target.shape[0]

# use lightgbm for regression
print('    Time elapsed %.0f sec'%(time.time()-start_time))

# specify your configurations as a dict
params = {
    'nthread': -1,
    'max_depth': 8,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'num_leaves': 31,
    'learning_rate': 0.25,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.06,
    'lambda_l2': 0.1,
    'verbose': -1
}

# do the training
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
        
    # train
    gbm = lgb.train(params, lgb_train, 1000, 
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=50, verbose_eval=100)
    oof_preds[valid_idx] = gbm.predict(valid_x, num_iteration=gbm.best_iteration)
    sub_preds[:] += gbm.predict(test_x, num_iteration=gbm.best_iteration) / folds.n_splits
    valid_idx += 1

# scale things back up
all_y *= tsy[:len_train]
oof_preds *= tsy[:len_train]
e = 2 * abs(all_y - oof_preds) / np.maximum( 0.1, abs(all_y)+abs(oof_preds) )
e = e.mean()
print('Full validation score %.4f' %e)

# Write submission file
pred = (sub_preds * tsy[len_train:] ).astype(np.float32)
out_df = pd.DataFrame({'id': ID.astype(np.int32), 'sales': pred})
out_df.to_csv('submission.csv', index=False)
print('    Time elapsed %.0f sec'%(time.time()-start_time))
