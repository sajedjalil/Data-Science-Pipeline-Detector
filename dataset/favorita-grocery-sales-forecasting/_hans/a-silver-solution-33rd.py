# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import date, timedelta
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from scipy import sparse as ssp

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print check_output(["ls", "../input"]).decode("utf8")

# Any results you write to the current directory are saved as output.

df_train = pd.read_csv(
    '../input/train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},  
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0}, 
    parse_dates=['date'], 
    skiprows=range(1, 66458909)
)

df_test = pd.read_csv(
    '../input/test.csv', usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=['date']  
).set_index(
    ['store_nbr', 'item_nbr', 'date'] 
)

items = pd.read_csv(
    '../input/items.csv',
).set_index('item_nbr')

df_2017 = df_train.loc[df_train.date>=pd.datetime(2017,1,1)] 
del df_train

promo_2017_train = df_2017.set_index(
    ['store_nbr', 'item_nbr', 'date'])[['onpromotion']].unstack(
        level=-1).fillna(False)
promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
promo_2017_test = df_test[['onpromotion']].unstack(level=-1).fillna(False)
promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
del promo_2017_test, promo_2017_train

df_2017 = df_2017.set_index(
    ['store_nbr', 'item_nbr', 'date'])[['unit_sales']].unstack(
        level=-1).fillna(0)
df_2017.columns = df_2017.columns.get_level_values(1)

items = items.reindex(df_2017.index.get_level_values(1))

def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def timeSeriesProcess(df_2017, t2017, label):
    X = pd.DataFrame()
    duration_list = [1, 3, 7, 14, 28, 60, 140]
    for d in duration_list:
        timespan = get_timespan(df_2017, t2017, d, d)
        X['{}_day_{}_2017'.format(label, d)] = timespan.mean(axis=1).values
        if d != 1:
            X['{}_day_{}_2017_max'.format(label,d)] = timespan.max(axis=1).values
            X['{}_day_{}_2017_min'.format(label,d)] = timespan.min(axis=1).values
            X['{}_day_{}_2017_var'.format(label,d)] = timespan.var(axis=1).values
            X['{}_day_{}_2017_skew'.format(label,d)] = timespan.skew(axis=1).values
            X['{}_day_{}_2017_kurt'.format(label,d)] = timespan.kurt(axis=1).values
            
            exp_sum = np.zeros(timespan.shape[0])
            for i in range(timespan.shape[1]):
                exp_sum += np.exp(-i/5) * timespan.iloc[:,i]
            X['{}_exp_moving_sum_{}'.format(label,d)] = exp_sum.values
    
    for idx in range(1,len(duration_list)):
        a = duration_list[idx-1]
        b = duration_list[idx]
        X['{}_day_{}sub{}_2017'.format(label, a,b)] = X['{}_day_{}_2017'.format(label, a)]                                                     - X['{}_day_{}_2017'.format(label, b)]
        
    for i in range(7):
        for j in [4, 10, 20]:
            timespan = get_timespan(df_2017, t2017, j*7-i, j, freq='7D')
            X['{}_mean_{}_dow{}_2017'.format(label, j, i)] = timespan.mean(axis=1).values
            
        date = t2017-timedelta(7-i)
        for m in [3,7,14,28,60,130]:
            X['{}_mean_{}_2017_{}_1'.format(label, m,i)]= get_timespan(
                df_2017, date, m, m).mean(axis=1).values
            X['{}_mean_{}_2017_{}_2'.format(label, m,i)]= get_timespan(
                df_2017, date-timedelta(7), m, m).mean(axis=1).values
    return X
    
def prepare_dataset(df_2017, t2017, promo_2017, is_train=True):
    X = pd.DataFrame({
        'store_nbr':df_2017.index.get_level_values(0),
        'item_nbr':df_2017.index.get_level_values(1),
        'unpromo_16aftsum_2017':(1-get_timespan(promo_2017, t2017, 0, 16)).sum(axis=1).values
    })
    duration_list = [1, 3, 7, 14, 28, 60, 140]
    for d in duration_list:       
        X['promo_{}_2017'.format(d)] = get_timespan(promo_2017, t2017, d, d).sum(axis=1).values
    
    for i in range(16):
        X['promo_{}'.format(i)] = promo_2017[t2017 + timedelta(days=i)].values.astype(np.uint8)
    
    X = pd.concat([X, timeSeriesProcess(df_2017, t2017, 'item')], axis=1)

    if is_train:
        y = df_2017[pd.date_range(t2017, periods=16)].values
        return X, y
    return X

num_training_weeks = 8
print('Preparing dataset...')
t2017 = date(2017, 5, 31) 
X_l, y_l = [], []
for i in range(num_training_weeks):
    print('training set ' + str(i) + ':')
    delta = timedelta(days = 7 * i)  
    X_tmp, y_tmp = prepare_dataset(
        df_2017, t2017 + delta, promo_2017
    )
    X_l.append(X_tmp)
    y_l.append(y_tmp)
X_train = pd.concat(X_l, axis=0, ignore_index=True)
y_train = np.concatenate(y_l, axis=0)  
del X_l, y_l

print('validation set:')
X_val, y_val = prepare_dataset(df_2017, date(2017, 7, 26), promo_2017)
print('testing set:')
X_test = prepare_dataset(df_2017, date(2017, 8, 16), promo_2017, is_train=False)

del X_train['item_nbr']
del X_val['item_nbr']
del X_test['item_nbr']

le = LabelEncoder()
items['family'] = le.fit_transform(items['family'])
X_train['family'] = pd.concat([items['family']] * num_training_weeks).values
X_val['family'] = items['family'].values
X_test['family'] = items['family'].values
X_train['class'] = pd.concat([items['class']] * num_training_weeks).values
X_val['class'] = items['class'].values
X_test['class'] = items['class'].values

store_info = pd.read_csv('../input/stores.csv', usecols=[0, 3, 4])
X_train = pd.merge(X_train, store_info, on='store_nbr', how='left')
X_val = pd.merge(X_val, store_info, on='store_nbr', how='left')
X_test = pd.merge(X_test, store_info, on='store_nbr', how='left')

cat_features = ['store_nbr','type','cluster','family','class']
num_features = [i for i in X_train.columns if i not in cat_features]
for col in cat_features:
    le = LabelEncoder()
    le.fit(pd.concat([X_train[col].drop_duplicates(), X_val[col].drop_duplicates(), X_test[col].drop_duplicates()]))
    X_train[col] = le.transform(X_train[col])
    X_val[col] = le.transform(X_val[col])
    X_test[col] = le.transform(X_test[col])
    
enc = OneHotEncoder()
enc.fit(pd.concat([X_train[cat_features],X_val[cat_features],X_test[cat_features]]))
X_train_cat = enc.transform(X_train[cat_features])
X_val_cat = enc.transform(X_val[cat_features])
X_test_cat = enc.transform(X_test[cat_features])

cat_count_features = []
for col in cat_features:
    d = pd.concat([X_train[col],X_val[col],X_test[col]]).value_counts().to_dict()
    X_train['%s_count'%col] = X_train[col].apply(lambda x:d.get(x,0))
    X_val['%s_count'%col] = X_val[col].apply(lambda x:d.get(x,0))   
    X_test['%s_count'%col] = X_test[col].apply(lambda x:d.get(x,0))
    cat_count_features.append('%s_count'%col)

X_train = ssp.hstack([X_train[num_features+cat_count_features].values,X_train_cat,]).tocsr()
X_val = ssp.hstack([X_val[num_features+cat_count_features].values,X_val_cat,]).tocsr()
X_test = ssp.hstack([X_test[num_features+cat_count_features].values,X_test_cat,]).tocsr()

print('Training and predicting models...')
params = {
    'num_leaves': 33,
    'objective': 'regression',
    'min_data_in_leaf': 1500,
    'learning_rate': 0.02,
    'feature_fraction': 0.7,
    'min_split_gain': 0,
    'metric': 'l2',
    'subsample': 0.9,
    'drop_rate': 0.1,
    'min_child_samples': 10,
    'min_child_weight': 150,
    'max_drop': 50,
    'boosting':'gbdt'
}

MAX_ROUNDS = 10000
val_pred = []
test_pred = []
seed_list = [1, 3, 5, 7]
for i in range(16):
    print('Step %d' % (i+1))
    val_res = np.zeros(X_val.shape[0])
    test_res = np.zeros(X_test.shape[0])
    dtrain = lgb.Dataset(X_train, label=y_train[:, i],
        weight=pd.concat([items['perishable']] * num_training_weeks) * 0.25 + 1 
    )
    dval = lgb.Dataset(X_val, label=y_val[:, i], reference=dtrain,
        weight=items['perishable'] * 0.25 + 1
    )
    
    for seed in seed_list:
        print('seed: %d' % seed)
        params['seed'] = seed
        bst = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS,
            valid_sets=dval, early_stopping_rounds=100, verbose_eval=100
        )
        val_res += bst.predict(X_val, num_iteration=bst.best_iteration) / len(seed_list)
        test_res += bst.predict(X_test, num_iteration=bst.best_iteration) / len(seed_list)

    val_pred.append(val_res)
    test_pred.append(test_res)

print('Validation mse:', mean_squared_error(y_val, np.array(val_pred).transpose(),
                                            sample_weight=items['perishable'] * 0.25 + 1))

print('Making submission...')
y_test = np.array(test_pred).transpose()
df_preds = pd.DataFrame(
    y_test, index=df_2017.index,
    columns=pd.date_range('2017-08-16', periods=16)
)
df_preds = df_preds.stack().to_frame('unit_sales')
df_preds.index.set_names(['store_nbr', 'item_nbr', 'date'], inplace=True)

submission = df_test[['id']].join(df_preds, how='left').fillna(0)
submission['unit_sales'] = np.clip(np.expm1(submission['unit_sales']), 0, 1000)
submission.to_csv('../submit/1-14lgb_item_cat_4fold_0.02.csv.gz',
                  float_format='%.4f', index=None,compression = 'gzip')