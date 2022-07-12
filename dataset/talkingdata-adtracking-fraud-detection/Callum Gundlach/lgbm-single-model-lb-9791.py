# BIG SHOUTOUT TO PRANAV FOR HIS MODEL
# Thanks to panjn for his helper methods

from sklearn.model_selection import train_test_split
import pandas as pd
import time
import numpy as np
import lightgbm as lgb
import os
os.environ['OMP_NUM_THREADS'] = '4'

max_rounds = 1000
early_stop = 50
opt_rounds = 680

output_file = 'lgbm_submit.csv'

path = "../input/"

dtypes = {
    'ip'		:'uint32',
    'app'		:'uint16',
	'device'	:'uint16',
	'os'		:'uint16',
	'channel'	:'uint16',
	'is_attributed'	:'uint8',
	'click_id'	:'uint32',
	}

print('Loading train.csv...')

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'is_attributed', 'click_time']
train_df = pd.read_csv(path + 'train.csv', skiprows=range(1,84903891), nrows=100000000, dtype=dtypes, usecols=train_cols)
#train_df = pd.read_csv(path + 'train.csv', dtype=dtypes, usecols=train_cols)

print('Load test.csv...')
test_cols = ['ip', 'app', 'device', 'os', 'click_time', 'channel', 'click_id']
test_df = pd.read_csv(path + "test.csv", dtype=dtypes, usecols=test_cols)


import gc

len_train = len(train_df)

print('Preprocessing...')

most_freq_hours_in_test_data = [4,5,9,10,13,14]
least_freq_hours_in_test_data = [6, 11, 15]

def add_counts(df, cols):
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(axis=0)+1),
                                     return_inverse=True, return_counts=True)
    df["_".join(cols)+"_count"] = counts[unqtags]

def add_next_click(df):
    D = 2**26
    df['category'] = (df['ip'].astype(str) + "_" + df['app'].astype(str) + "_" + df['device'].astype(str) \
                      + "_" + df['os'].astype(str)).apply(hash) % D
    click_buffer = np.full(D, 3000000000, dtype=np.uint32)
    df['epochtime'] = df['click_time'].astype(np.int64) // 10 ** 9
    next_clicks = []
    for category, time in zip(reversed(df['category'].values), reversed(df['epochtime'].values)):
        next_clicks.append(click_buffer[category] - time)
        click_buffer[category] = time
    del click_buffer
    df['next_click'] = list(reversed(next_clicks))
    df.drop(['category', 'epochtime'], axis=1, inplace=True)

def preproc_data(df):
    
    #Extrace date info
    df['click_time']= pd.to_datetime(df['click_time'])
    df['hour'] = df['click_time'].dt.hour.astype('uint8')
    df['day'] = df['click_time'].dt.day.astype('uint8')
    df['wday'] = df['click_time'].dt.dayofweek.astype('uint8')
    gc.collect()

    #Groups
    df['in_test_hh'] = ( 3
	    		 - 2 * df['hour'].isin( most_freq_hours_in_test_data )
			 - 1 * df['hour'].isin( least_freq_hours_in_test_data )).astype('uint8')

    print('Adding next_click...')
    add_next_click(df)

    print('Grouping...')
    
    add_counts(df, ['ip'])
    add_counts(df, ['os', 'device'])
    add_counts(df, ['os', 'app', 'channel'])

    add_counts(df, ['ip', 'device'])
    add_counts(df, ['app', 'channel'])

    add_counts(df, ['ip', 'wday', 'in_test_hh'])
    add_counts(df, ['ip', 'wday', 'hour'])
    add_counts(df, ['ip', 'os', 'wday', 'hour'])
    add_counts(df, ['ip', 'app', 'wday', 'hour'])
    add_counts(df, ['ip', 'device', 'wday', 'hour'])
    add_counts(df, ['ip', 'app', 'os'])
    add_counts(df, ['wday', 'hour', 'app'])
    

    df.drop(['ip', 'day', 'click_time'], axis=1, inplace=True )
    gc.collect()

    print( df.info() )

    return df

y = train_df.is_attributed.values

submit = pd.DataFrame()
submit['click_id'] = test_df['click_id']

train_len = len(train_df)
common_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
train_df = pd.concat([train_df[common_cols], test_df[common_cols]])

train_df = preproc_data(train_df)

test_df = train_df.iloc[train_len:]
train_df = train_df.iloc[:train_len]

gc.collect()

metrics = 'auc'
lgb_params = {
	'boosting_type': 'gbdt',
	'objective': 'binary',
	'metric': metrics,
	'learning_rate': .1,
	'num_leaves': 7,
	'max_depth': 4,
	'min_child_samples': 100,
	'max_bin': 100,
	'subsample': 0.7,
	'subsample_freq': 1,
	'colsample_bytree': 0.7,
	'min_child_weight': 0,
	'min_split_gain': 0,
	'nthread': 4,
	'verbose': 1,
	'scale_pos_weight': 99.7
	#'scale_pos_weight': 400
}

target = 'is_attributed'

inputs = list(set(train_df.columns) - set([target]))  
cat_vars = ['app', 'device', 'os', 'channel', 'hour', 'wday']

train_df, val_df = train_test_split(train_df, train_size=.95, shuffle=False)
y_train, y_val = train_test_split(y, train_size=.95, shuffle=False)

print('Train size:', len(train_df))
print('Valid size:', len(val_df))

gc.collect()

print('Training...')

num_boost_round=max_rounds
early_stopping_rounds=early_stop

xgtrain = lgb.Dataset(train_df[inputs].values, label=y_train,
		      feature_name=inputs,
		      categorical_feature=cat_vars)
del train_df
gc.collect()

xgvalid = lgb.Dataset(val_df[inputs].values, label=y_val,
		      feature_name=inputs,
		      categorical_feature=cat_vars)
del val_df
gc.collect()

evals_results = {}

model = lgb.train(lgb_params,
		  xgtrain,
		  valid_sets= [xgvalid],
		  valid_names=['valid'],
		  evals_result=evals_results,
		  num_boost_round=num_boost_round,
		  early_stopping_rounds=early_stopping_rounds,
		  verbose_eval=1,
		  feval=None)
n_estimators = model.best_iteration

print('\nModel Info:')
print('n_estimators:', n_estimators)
print(metrics+':', evals_results['valid'][metrics][n_estimators-1])

del xgvalid
del xgtrain
gc.collect()


print('Predicting...')
submit['is_attributed'] = model.predict(test_df[inputs])

print('Creating:', output_file)
submit.to_csv(output_file, index=False, float_format='%.9f')
print('Done!')
