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
import gc
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score


# read raw data
start_time = time.time()
input_dir = os.path.join(os.pardir, 'input')
print('  Loading data...')
train_df = pd.read_csv(os.path.join(input_dir, 'train.csv'), nrows=None)
test_df = pd.read_csv(os.path.join(input_dir, 'test.csv'), nrows=None)
print('    Time elapsed %.0f sec'%(time.time()-start_time))
print('Using %d prediction variables'%(train_df.shape[1]-2))
print('  Pre-processing data...')


#df2 = train_df.groupby('idhogar').agg(['mean','min','max','std'])
#train_df = train_df.join(df2, how='left', on='idhogar')


# only select hoh for training. This also corrects target values.
train_df = train_df[train_df.parentesco1==1]


# Merge test/train datasets into a single one and separate unneeded columns
target = train_df.pop('Target')
len_train = len(train_df)
merged_df = pd.concat([train_df, test_df])
print( merged_df.shape )
del test_df, train_df
gc.collect()
print('  Time elapsed %.0f sec'%(time.time()-start_time))


# drop all squares(and some others)
merged_df.drop(['idhogar','SQBescolari','SQBage','SQBhogar_total','SQBedjefe','SQBhogar_nin','SQBovercrowding','SQBdependency','SQBmeaned','agesq'], axis=1, inplace=True)

# pop/drop keys
key = merged_df.pop('Id')
key = key[len_train:]


# fix yes/no
merged_df.loc[merged_df.dependency == 'yes', 'dependency'] = 1
merged_df.loc[merged_df.dependency == 'no', 'dependency'] = 0

merged_df.loc[merged_df.edjefe == 'yes', 'edjefe'] = 1
merged_df.loc[merged_df.edjefe == 'no', 'edjefe'] = 0

merged_df.loc[merged_df.edjefa == 'yes', 'edjefa'] = 1
merged_df.loc[merged_df.edjefa == 'no', 'edjefa'] = 0

merged_df['dependency'] = merged_df['dependency'].astype(np.float32)
merged_df['edjefe'] = merged_df['edjefe'].astype(np.int8)
merged_df['edjefa'] = merged_df['edjefa'].astype(np.int8)


# use lightgbm for regression
print(' start training...\n    Time elapsed %.0f sec'%(time.time()-start_time))
# specify config as a dict
params = {
    'max_depth': 8,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 4,
    'metric': '',
    'num_leaves': 7,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0,
    'lambda_l2': 1,
    'verbose': -1
}

# do the training
target = target - 1 # 0 to 3
num_folds = 5
test_x = merged_df[len_train:]
oof_preds = np.zeros([len_train])
sub_preds = np.zeros([test_x.shape[0],4])
folds = KFold(n_splits=num_folds, shuffle=True, random_state=4564)
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(merged_df[:len_train])):
    lgb_train = lgb.Dataset(merged_df.iloc[train_idx], target.iloc[train_idx])
    lgb_valid = lgb.Dataset(merged_df.iloc[valid_idx], target.iloc[valid_idx])
        
    # train
    gbm = lgb.train(params, lgb_train, 5000, valid_sets=[lgb_train, lgb_valid], early_stopping_rounds=100, verbose_eval=1000)
    pr1 = gbm.predict(merged_df.iloc[valid_idx], num_iteration=gbm.best_iteration)
    pr2 = gbm.predict(test_x, num_iteration=gbm.best_iteration)
    pr1 = pr1 * np.array([1.2, 0.87, 1.135, 0.41]) # scale to producte the same frequency distribution in predicted values
    pr2 = pr2 * np.array([1.2, 0.87, 1.135, 0.41])
    oof_preds[valid_idx] = pr1.argmax(axis=1)
    sub_preds += pr2 / folds.n_splits
    valid_idx += 1
sub_preds = sub_preds.argmax(axis=1)
e = f1_score(target, oof_preds, average='macro')
print('Full validation score %.6f' %e)
print('    Time elapsed %.0f sec'%(time.time()-start_time))


# Write submission file
out_df = pd.DataFrame({'Id': key})
out_df['Target'] = sub_preds.astype(np.float32)

# round/cap/floor
out_df['Target'] = (out_df['Target'] + 0.5 + 1).astype(np.int8) # turn into 1-4
out_df['Target'] = np.maximum(out_df['Target'], 1)
out_df['Target'] = np.minimum(out_df['Target'], 4)

out_df.to_csv('submission.csv', index=False)
print('    Time elapsed %.0f sec'%(time.time()-start_time))
