# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import time
import gc
import sys
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold


def reduce_size(merged_df):
    print('      Starting size is %d Mb'%(sys.getsizeof(merged_df)/1024/1024))
    print('      Columns: %d'%(merged_df.shape[1]))
    feats = merged_df.columns[merged_df.dtypes == 'float64']
    for feat in feats:
        merged_df[feat] = merged_df[feat].astype('float32')

    feats = merged_df.columns[merged_df.dtypes == 'int16']
    for feat in feats:
        mm = np.abs(merged_df[feat]).max()
        if mm < 126:
            merged_df[feat] = merged_df[feat].astype('int8')

    feats = merged_df.columns[merged_df.dtypes == 'int32']
    for feat in feats:
        mm = np.abs(merged_df[feat]).max()
        if mm < 126:
            merged_df[feat] = merged_df[feat].astype('int8')
        elif mm < 30000:
            merged_df[feat] = merged_df[feat].astype('int16')

    feats = merged_df.columns[merged_df.dtypes == 'int64']
    for feat in feats:
        mm = np.abs(merged_df[feat]).max()
        if mm < 126:
            merged_df[feat] = merged_df[feat].astype('int8')
        elif mm < 30000:
            merged_df[feat] = merged_df[feat].astype('int16')
        elif mm < 2000000000:
            merged_df[feat] = merged_df[feat].astype('int32')
    print('      Ending size is %d Mb'%(sys.getsizeof(merged_df)/1024/1024))
    return merged_df


# read data
print(' Start')
start_time = time.time()
input_dir = os.path.join(os.pardir, 'input')
train_df = pd.read_csv(os.path.join(input_dir, 'train_V2.csv'), nrows=None)


# drop 1/2 of training matches
train_df.sort_values('matchId',inplace=True)
a = train_df['matchId'].unique()
b = int(a.shape[0]/3) # keep top 1/3 of matches to control RAM size and runtime
a = a[0:b]
train_df = train_df[train_df['matchId'].isin(a)].reset_index()
train_df.drop('index', axis=1, inplace=True)
del a, b
gc.collect()

# now read test data
print('  Now read test dataset. Time elapsed %.0f sec'%(time.time()-start_time))
test_df = pd.read_csv(os.path.join(input_dir, 'test_V2.csv'), nrows=None) # test only!!!!!


# Merge test/train datasets into a single one and separate unneeded columns
print(' Merge datasets. Time elapsed %.0f sec'%(time.time()-start_time))
target = train_df.pop('winPlacePerc')
target.fillna(0.5, inplace=True) # fix broken targets
len_train = len(train_df)
merged_df = pd.concat([train_df, test_df])
print( merged_df.shape )
del test_df, train_df
gc.collect()
merged_df['matchType'] = merged_df['matchType'].astype('category')
merged_df = reduce_size(merged_df) # reduce size


# add some columns
merged_df['skill'] = merged_df['headshotKills'] + 0.01 * merged_df['longestKill'] - merged_df['teamKills']/(merged_df['kills']+1)

merged_df['hsRatio'] = merged_df['headshotKills'] / merged_df['kills']
merged_df['hsRatio'].fillna(0, inplace=True)

merged_df['distance'] = (merged_df['walkDistance'] + 0.4 * merged_df['rideDistance'] + merged_df['swimDistance'])/merged_df['matchDuration']

merged_df['boostsRatio'] = merged_df['boosts']**2 / merged_df['walkDistance']**0.5
merged_df['boostsRatio'].fillna(0, inplace=True)
merged_df['boostsRatio'].replace(np.inf, 0, inplace=True)

merged_df['healsRatio'] = merged_df['heals'] / merged_df['matchDuration']**0.1
merged_df['healsRatio'].fillna(0, inplace=True)
merged_df['healsRatio'].replace(np.inf, 0, inplace=True)

merged_df['killsRatio'] = merged_df['kills'] / merged_df['matchDuration']**0.1
merged_df['killsRatio'].fillna(0, inplace=True)
merged_df['killsRatio'].replace(np.inf, 0, inplace=True)


# add rank by match
cc0 = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'revives', 'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
       'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints', 'matchDuration','rankPoints',
       'skill', 'hsRatio', 'distance', 'boostsRatio', 'healsRatio', 'killsRatio']
print(' Add rank by match. Time elapsed %.0f sec'%(time.time()-start_time))
df2 = merged_df.groupby(['Id','matchId'])[cc0].agg('max')
df2.fillna(0.5, inplace=True) # in some matches roadKills are all 0, so /mean is NAN
df2 = df2.groupby('matchId')[cc0].rank(pct=True)


# redefine rank to begin from 0 - part 1
dft = pd.DataFrame(merged_df.groupby('matchId')['boosts'].agg('size'))
dft.columns=['cc']
dft.reset_index(inplace=True)
df2.reset_index(inplace=True)
df2 = df2.merge(dft, how='left', on='matchId')
ccc = df2.drop(['Id','matchId','cc'], axis=1).columns
for c in ccc:
    df2[c] = (df2['cc']*df2[c]-1)/(df2['cc']-1)
df2.fillna(0.5, inplace=True)
df2.drop('cc', axis=1, inplace=True)
del dft, ccc
gc.collect()


df2 = reduce_size(df2) # reduce size
merged_df = merged_df.merge(df2, how='left', on=['Id','matchId'], suffixes=['','_s'])
del df2
gc.collect()
# add scaled to median by match
print(' Add scaled to mean by match. Time elapsed %.0f sec'%(time.time()-start_time))
df2 = merged_df.groupby('matchId')[cc0].agg('median')
df2.columns = [c[1]+'_mean' for c in enumerate(df2.columns)]
df2 = reduce_size(df2) # reduce size
merged_df = merged_df.merge(df2, how='left', on='matchId')
del df2
gc.collect()
for col in cc0:
    merged_df[col+'_s2'] = (merged_df[col] / merged_df[col+'_mean']).astype('float32')
    merged_df.drop(col+'_mean', axis=1, inplace=True)
# add max of everything by group+match
cc0 = ['assists_s', 'boosts_s', 'damageDealt_s', 'DBNOs_s', 'headshotKills_s', 'heals_s', 'killPlace_s', 'killPoints_s', 'kills_s',
       'killStreaks_s', 'longestKill_s', 'revives_s', 'rideDistance_s', 'roadKills_s', 'swimDistance_s', 'teamKills_s',
       'vehicleDestroys_s', 'walkDistance_s', 'weaponsAcquired_s', 'winPoints_s', 'matchDuration_s','rankPoints_s',
       'skill_s', 'hsRatio_s', 'distance_s', 'boostsRatio_s', 'healsRatio_s', 'killsRatio_s',
       'assists_s2', 'boosts_s2', 'damageDealt_s2', 'DBNOs_s2', 'headshotKills_s2', 'heals_s2', 'killPlace_s2', 'killPoints_s2', 'kills_s2',
       'killStreaks_s2', 'longestKill_s2', 'revives_s2', 'rideDistance_s2', 'roadKills_s2', 'swimDistance_s2', 'teamKills_s2',
       'vehicleDestroys_s2', 'walkDistance_s2', 'weaponsAcquired_s2', 'winPoints_s2', 'matchDuration_s2','rankPoints_s2',
       'skill_s2', 'hsRatio_s2', 'distance_s2', 'boostsRatio_s2', 'healsRatio_s2', 'killsRatio_s2'
]
print(' Add max of everything by group+match. Time elapsed %.0f sec'%(time.time()-start_time))
df2 = merged_df.groupby(['groupId','matchId'])[cc0].agg('max')
df2 = reduce_size(df2) # reduce size
merged_df = merged_df.merge(df2, how='left', on=['groupId','matchId'], suffixes=['','_max'])
# add max rank of everything by group+match
print(' Add max rank of everything by group+match. Time elapsed %.0f sec'%(time.time()-start_time))
df2.fillna(0.5, inplace=True) # in some matches roadKills are all 0, so /mean is NAN
df2 = df2.groupby('matchId')[cc0].rank(pct=True)


# redefine rank to begin from 0 - part 2
dft = pd.DataFrame(merged_df.groupby(['groupId','matchId'])['boosts'].agg('size'))
dft.columns=['cc']
dft.reset_index(inplace=True)
dft = pd.DataFrame(dft.groupby('matchId')['cc'].agg('size'))
dft.columns=['cc']
dft.reset_index(inplace=True)
df2.reset_index(inplace=True)
df2 = df2.merge(dft, how='left', on='matchId')
ccc = df2.drop(['groupId','matchId','cc'], axis=1).columns
for c in ccc:
    df2[c] = (df2['cc']*df2[c]-1)/(df2['cc']-1)
df2.fillna(0.5, inplace=True)
df2.drop('cc', axis=1, inplace=True)
del dft, ccc
gc.collect()


df2 = reduce_size(df2) # reduce size
merged_df = merged_df.merge(df2, how='left', on=['groupId','matchId'], suffixes=['','_maxrank'])
del df2
gc.collect()
# pop IDs
idd = merged_df.pop('Id')
groupIdd = merged_df.pop('groupId')
matchIdd = merged_df.pop('matchId')


# drop features that have low importance
merged_df.drop(['matchDuration_s2','vehicleDestroys','matchDuration_s2_max','roadKills','teamKills','headshotKills','revives','assists',
                'DBNOs','killStreaks','matchDuration_s2_maxrank','vehicleDestroys_s2',
                'healsRatio_s2_maxrank','killsRatio_s2_maxrank'], axis=1, inplace=True)


# use lightgbm for regression
print(' Start training. Time elapsed %.0f sec'%(time.time()-start_time))
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': 'l1',
    'num_leaves': 511,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 10,
    'lambda_l2': 10
}
num_runs = 1
num_folds = 2
np.random.seed(seed=1567215968)
randoms = np.random.randint(2000000000,high=None,size=num_runs)
oof_preds = np.zeros([num_runs,len_train])
sub_preds = np.zeros([num_runs,merged_df.shape[0]-len_train])
groups = np.array(idd.iloc[:len_train].astype('category')) # idd, groupIdd, matchIdd
for runs in range(num_runs):
    print('  Starting run %d/%d. Time elapsed %.0f sec'%(runs+1,num_runs,time.time()-start_time))
    folds = GroupKFold(n_splits=num_folds)
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X=merged_df.iloc[:len_train], y=target, groups=groups)):
        print('  Starting fold/run %d/%d. Time elapsed %.0f sec'%(n_fold+1,runs+1,time.time()-start_time))
        # drop single matches from training set
        dd = merged_df.iloc[:len_train]['maxPlace']
        train_idx2 = np.array((dd[train_idx]>75).index)
        lgb_train = lgb.Dataset(merged_df.iloc[train_idx2], target.iloc[train_idx2])
        lgb_valid = lgb.Dataset(merged_df.iloc[valid_idx], target.iloc[valid_idx])
            
        # train
        gbm = lgb.train(params, lgb_train, 4000, valid_sets=[lgb_train, lgb_valid], early_stopping_rounds=50, verbose_eval=100)
        print('    Time elapsed %.0f sec'%(time.time()-start_time))
        oof_preds[runs,valid_idx] = np.minimum(1,np.maximum(0,gbm.predict(merged_df.iloc[valid_idx], num_iteration=gbm.best_iteration)))
        sub_preds[runs,:] += np.minimum(1,np.maximum(0,gbm.predict(merged_df.iloc[len_train:], num_iteration=gbm.best_iteration))) / folds.n_splits
oof_preds = oof_preds.mean(axis=0)
sub_preds = sub_preds.mean(axis=0)
e = abs(target - oof_preds).mean()
print('Full validation score by player %.5f *******************' %e)



# now train only maxPlace>75, assuming single players.
cols2a = ['boosts', 'damageDealt', 'heals', 'killPlace', 'killPoints', 'kills', 'longestKill', 'matchDuration', 'matchType', 'maxPlace', 'numGroups', 'rankPoints', 'rideDistance',
           'swimDistance', 'walkDistance', 'weaponsAcquired', 'winPoints', 'skill', 'hsRatio', 'distance', 'boostsRatio', 'healsRatio', 'killsRatio', 'assists_s', 'boosts_s',
           'damageDealt_s', 'DBNOs_s', 'headshotKills_s', 'heals_s', 'killPlace_s', 'killPoints_s', 'kills_s', 'killStreaks_s', 'longestKill_s', 'revives_s', 'rideDistance_s',
           'roadKills_s', 'swimDistance_s', 'teamKills_s', 'vehicleDestroys_s', 'walkDistance_s', 'weaponsAcquired_s', 'winPoints_s', 'rankPoints_s', 'skill_s',
           'hsRatio_s', 'distance_s', 'boostsRatio_s', 'healsRatio_s', 'killsRatio_s', 'assists_s2', 'boosts_s2', 'damageDealt_s2', 'headshotKills_s2', 'heals_s2',
           'killPlace_s2', 'killPoints_s2', 'kills_s2', 'killStreaks_s2', 'longestKill_s2', 'rideDistance_s2', 'roadKills_s2', 'swimDistance_s2', 'teamKills_s2',
           'walkDistance_s2', 'weaponsAcquired_s2', 'winPoints_s2', 'rankPoints_s2', 'skill_s2', 'hsRatio_s2', 'distance_s2', 'boostsRatio_s2', 'healsRatio_s2', 'killsRatio_s2']
dd = merged_df.iloc[:len_train]
dd2 = dd[dd['maxPlace']>75]
y2 = target[dd['maxPlace']>75]
dd2 = dd2[cols2a]
# now test set
dd = merged_df.iloc[len_train:]
dd2a = dd[dd['maxPlace']>75]
dd2a = dd2a[cols2a]
randoms = np.random.randint(2000000000,high=None,size=num_runs)
oof_preds2 = np.zeros([num_runs,dd2.shape[0]])
sub_preds2 = np.zeros([num_runs,dd2a.shape[0]])
for runs in range(num_runs):
    print('  Starting run %d/%d. Time elapsed %.0f sec'%(runs+1,num_runs,time.time()-start_time))
    folds = KFold(n_splits=num_folds, shuffle=True, random_state=randoms[runs])
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X=dd2)):
        print('  Starting fold/run %d/%d. Time elapsed %.0f sec'%(n_fold+1,runs+1,time.time()-start_time))
        lgb_train = lgb.Dataset(dd2.iloc[train_idx], y2.iloc[train_idx])
        lgb_valid = lgb.Dataset(dd2.iloc[valid_idx], y2.iloc[valid_idx])
            
        # train
        gbm = lgb.train(params, lgb_train, 5000, valid_sets=[lgb_train, lgb_valid], early_stopping_rounds=50, verbose_eval=100)
        print('    Time elapsed %.0f sec'%(time.time()-start_time))
        oof_preds2[runs,valid_idx] = np.minimum(1,np.maximum(0,gbm.predict(dd2.iloc[valid_idx], num_iteration=gbm.best_iteration)))
        sub_preds2[runs,:] += np.minimum(1,np.maximum(0,gbm.predict(dd2a, num_iteration=gbm.best_iteration))) / folds.n_splits
oof_preds2 = oof_preds2.mean(axis=0)
sub_preds2 = sub_preds2.mean(axis=0)
e = abs(y2 - oof_preds2).mean()

idd2 = idd[:len_train]
mp2 = merged_df['maxPlace']
mp2 = mp2[:len_train]
pred1a_train = pd.DataFrame({'Id': idd2[mp2>75]})
pred1a_train['pred1a'] = oof_preds2

idd2 = idd[len_train:]
mp2 = merged_df['maxPlace']
mp2 = mp2[len_train:]
pred1a_test = pd.DataFrame({'Id': idd2[mp2>75]})
pred1a_test['pred1a'] = sub_preds2
print('Full validation score by player %.5f *******************' %e)


# combine by group/match
# add columns
merged_df['matchId'] = matchIdd
merged_df['groupId'] = groupIdd
merged_df['Id'] = idd
merged_df['pred1'] = np.concatenate([oof_preds,sub_preds])
merged_df['target'] = np.concatenate([target,np.zeros(sub_preds.shape[0])])
merged_df['one'] = np.concatenate([np.ones(oof_preds.shape[0]),np.zeros(sub_preds.shape[0])])


# add rank of pred1 by match
print(' Add rank of pred1 by match. Time elapsed %.0f sec'%(time.time()-start_time))
df2 = merged_df.groupby(['Id','matchId'])['pred1'].agg('max')
df2.fillna(0.5, inplace=True)
df2 = pd.DataFrame(df2)
df2.columns = ['pred1']
df2 = df2.groupby('matchId')['pred1'].rank(pct=True)
df2 = pd.DataFrame(df2)
df2.columns = ['pred1_rr']


# redefine rank to begin from 0 - part 3
dft = pd.DataFrame(df2.groupby('matchId').agg('size'))
dft.columns=['cc']
dft.reset_index(inplace=True)
df2.reset_index(inplace=True)
df2 = df2.merge(dft, how='left', on='matchId')
df2['pred1_rr'] = (df2['cc']*df2['pred1_rr']-1)/(df2['cc']-1)
df2.fillna(0.5, inplace=True)
df2.drop('cc', axis=1, inplace=True)
del dft
gc.collect()


df2 = reduce_size(df2) # reduce size
merged_df = merged_df.merge(df2, how='left', on=['Id','matchId'], suffixes=['','_rr'])
del df2
gc.collect()


# group by groupId+matchId for second pass
aggregations = {
    'target': ['mean'],
    'pred1': ['median','mean','min','max','std','sum'],
    'pred1_rr': ['median','mean','min','max','std','size'],
    'maxPlace':['mean'],
    'one': ['max']
}
print('    Start grouping. Time elapsed %.0f sec'%(time.time()-start_time))
df2 = merged_df.groupby(['groupId','matchId']).agg(aggregations)
print('    Finished grouping. Time elapsed %.0f sec'%(time.time()-start_time))
df2.reset_index(inplace=True)
df2.columns = [c[1][0]+c[1][1]for c in enumerate(df2.columns)]
df2 = reduce_size(df2) # reduce size


# fit part 2
x_cols = ['pred1median','pred1mean','pred1min','pred1max','pred1std','pred1sum',
          'maxPlacemean','pred1_rrmedian','pred1_rrmean','pred1_rrmin','pred1_rrmax','pred1_rrstd','pred1_rrsize'
          ]
randoms = np.random.randint(2000000000,high=None,size=num_runs)
df2a = df2[df2['onemax']==1]
x = df2a[x_cols].copy()
y = df2a['targetmean']
groups = np.array(df2a['matchId'].astype('category')) # 'groupId', 'matchId'
del df2a
gc.collect()
df2b = df2[df2['onemax']==0]
x_test = df2b[x_cols].copy()
del df2b
gc.collect()
oof_preds = np.zeros([num_runs,x.shape[0]])
sub_preds = np.zeros([num_runs,x_test.shape[0]])
for runs in range(num_runs):
    print('  Starting run %d/%d. Time elapsed %.0f sec'%(runs+1,num_runs,time.time()-start_time))
    folds = GroupKFold(n_splits=num_folds)
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X=x, y=y, groups=groups)):
        print('  Starting fold/run %d/%d. Time elapsed %.0f sec'%(n_fold+1,runs+1,time.time()-start_time))
        lgb_train = lgb.Dataset(x.iloc[train_idx], y.iloc[train_idx],weight=x['pred1_rrsize'].iloc[train_idx]) # include weight - count of players per group
        lgb_valid = lgb.Dataset(x.iloc[valid_idx], y.iloc[valid_idx],weight=x['pred1_rrsize'].iloc[valid_idx])

        # train
        gbm = lgb.train(params, lgb_train, 2000, valid_sets=[lgb_train, lgb_valid], early_stopping_rounds=50, verbose_eval=100)
        print('    Time elapsed %.0f sec'%(time.time()-start_time))
        oof_preds[runs,valid_idx] = np.minimum(1,np.maximum(0,gbm.predict(x.iloc[valid_idx], num_iteration=gbm.best_iteration)))
        sub_preds[runs,:] += np.minimum(1,np.maximum(0,gbm.predict(x_test, num_iteration=gbm.best_iteration))) / folds.n_splits
oof_preds = oof_preds.mean(axis=0)
sub_preds = sub_preds.mean(axis=0)
e = ((x['pred1_rrsize']*abs(y - oof_preds)).sum())/x['pred1_rrsize'].sum()
del x, x_test, y
gc.collect()
print('Weighted validation score by group %.5f *******************' %e)

# apply result of part 2 back to source data
# to train data
out_df_t = pd.DataFrame({'Id': idd[:len_train]})
out_df_t['groupId'] = groupIdd[:len_train]
out_df_t['matchId'] = matchIdd[:len_train]
out_df_t['target'] = target

df3_t = df2[['groupId','matchId','maxPlacemean']]
df3_t = df3_t[df2['onemax']==1]
df3_t['pred3'] = oof_preds

# add pred1a
out_df_t = out_df_t.merge(pred1a_train, how='left', on='Id') # add pred1a
df3a_t = out_df_t.groupby(['groupId','matchId'])['pred1a'].agg('mean')
df3a_t = pd.DataFrame(df3a_t)
df3a_t.reset_index(inplace=True)
df3_t = df3_t.merge(df3a_t, how='left', on=['groupId','matchId'])
df3_t['pred1a'].fillna(2, inplace=True)
df3_t['pred3'] = df3_t['pred3']*(df3_t['pred1a']==2)+df3_t['pred1a']*(df3_t['pred1a']<2)

df3_t = df3_t.sort_values(['matchId','pred3'],ascending=True)

# rank groups in each match
aggregations = {
    'groupId': 'size',
    'maxPlacemean': 'mean'
}
df4_t = df3_t.groupby('matchId').agg(aggregations)
df4_t.reset_index()
l0 = list()
for k,l in zip(df4_t['maxPlacemean'],df4_t['groupId']):
    if l == 1:
        l0.append(0.5)
    else:
        l1 = np.arange(0,k) # long list
        # drop some (k-l) from the middle
        if k > l:
            l2 = int(l/2)+np.arange(0,k-l) # drop list
            l1 = [x for x in l1 if x not in l2]
        l1 = list(np.array(l1)/(k-1))
        l0.extend(l1)
df3_t['pred4'] = l0
df3_t['pred4'] = np.round(df3_t['pred4'],4)

out_df_t = out_df_t.merge(df3_t, how='left', on=['groupId','matchId'])
df4_t = pd.DataFrame(df4_t)
out_df_t = out_df_t.merge(df4_t, how='left', on='matchId')
out_df_t = out_df_t.sort_values(['matchId','target'],ascending=True)
e = abs(out_df_t['target'] - out_df_t['pred3']).mean()
print('Full validation score v2 by player %.5f *******************' %e)
out_df_t.drop('pred3', axis=1, inplace=True)
e = abs(out_df_t['target'] - out_df_t['pred4']).mean()
print('Full validation score v3 by player %.5f *******************' %e)


# apply result of part 2 back to source data
# Write submission file
out_df = pd.DataFrame({'Id': idd[len_train:]})
out_df['groupId'] = groupIdd[len_train:]
out_df['matchId'] = matchIdd[len_train:]

df3 = df2[['groupId','matchId','maxPlacemean']]
df3 = df3[df2['onemax']==0]
df3['winPlacePerc'] = sub_preds

# add pred1a
out_df = out_df.merge(pred1a_test, how='left', on='Id') # add pred1a
df3a = out_df.groupby(['groupId','matchId'])['pred1a'].agg('mean')
df3a = pd.DataFrame(df3a)
df3a.reset_index(inplace=True)
df3 = df3.merge(df3a, how='left', on=['groupId','matchId'])
df3['pred1a'].fillna(2, inplace=True)
df3['winPlacePerc'] = df3['winPlacePerc']*(df3['pred1a']==2)+df3['pred1a']*(df3['pred1a']<2)

df3 = df3.sort_values(['matchId','winPlacePerc'],ascending=True)

# rank groups in each match
aggregations = {
    'groupId': 'size',
    'maxPlacemean': 'mean'
}
df4 = df3.groupby('matchId').agg(aggregations)
df4.reset_index()
l0 = list()
for k,l in zip(df4['maxPlacemean'],df4['groupId']):
    if l == 1:
        l0.append(0.5)
    else:
        l1 = np.arange(0,k) # long list
        # drop some (k-l) from the middle
        if k > l:
            l2 = int(l/2)+np.arange(0,k-l) # drop list
            l1 = [x for x in l1 if x not in l2]
        l1 = list(np.array(l1)/(k-1))
        l0.extend(l1)
df3['pred4'] = l0
df3['pred4'] = np.round(df3['pred4'],4)

out_df2 = out_df.merge(df3, how='left', on=['groupId','matchId'])
out_df2.drop(['pred1a_x','pred1a_y','groupId', 'matchId','winPlacePerc','maxPlacemean'],axis=1,inplace=True)
out_df2.columns = ['Id','winPlacePerc']

out_df2.to_csv('submission.csv', index=False)
print('    Time elapsed %.0f sec'%(time.time()-start_time))