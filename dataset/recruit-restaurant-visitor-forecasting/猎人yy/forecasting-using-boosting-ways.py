# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 19:48:02 2018

@author: yuan
"""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn import preprocessing, metrics, ensemble, neighbors
from xgboost import XGBRegressor
import copy

# fea1:Consecutive_hol
def get_consecutive_hol(hol):
    hol['Consecutive_holidays'] = 0
    i = 0
    while i < len(hol):
        count = 0
        # 如果某天为周一或周五，且holiday_flg=1，则该天连同周末赋值为1
        if i > 0 and (hol.iloc[i, 1] == 'Friday' and hol.iloc[i, 2] == 1) and (hol.iloc[i-1, 2] == 0):
            hol.iloc[i,3] = 1
            if i+1 < len(hol):
                hol.iloc[i+1,3] = 1
                if i+2 < len(hol):
                    hol.iloc[i+2,3] = 1
    
        if i < len(hol)-1 and (hol.iloc[i, 1] == 'Monday' and hol.iloc[i, 2] == 1) and hol.iloc[i+1,2] == 0:
            hol.iloc[i,3] = 1
            if i-1 >= 0:
                hol.iloc[i-1,3] = 1
                if i-2 >= 0:
                    hol.iloc[i-2,3] = 1
    
        # 如果有一天是holiday,且这一天在周一至周五之间
        if hol.iloc[i,1] not in {'Saturday','Sunday'} and hol.iloc[i, 2] == 1:
            j = copy.deepcopy(i) + 1
            count = 1
            while j <= len(hol)-1 and (hol.iloc[j,2] == 1 or hol.iloc[j,1] in {'Saturday','Sunday'}):
                count += 1
                j += 1
                if count < 3:
                    pass
                elif count == 3:
                    hol.iloc[i:i+count,3] = 1
                elif count > 3:
                    hol.iloc[i:i+count,3] = 2
    
        if count == 0:
            i += 1
        else:
            i += count

    i = 0
    while i < len(hol):
        # 如果某天在两个小长假之间，那么赋值为0.5
        if hol.iloc[i,3] in {1,2,0.5}:
            pass
        if hol.iloc[i,3] == 0:
            if i > 0 and i+1 < len(hol) and hol.iloc[i-1,3] in {1,2} and hol.iloc[i+1,3] in {1,2}:
                hol.iloc[i,3] = 0.5
        i += 1
    
    # 下面四句是把既是holiday且又是周末的holiday_flg赋值为0
    wkend_holidays = hol.apply((lambda x:(x.day_of_week=='Sunday' or x.day_of_week=='Saturday') and x.holiday_flg==1), axis=1)
    hol.loc[wkend_holidays,'holiday_flg'] = 0


# fea2:diff_vr/rv_sum
def get_diff_vr_rv_sum(data, hol):
    mean_lst = []
    for df in data:
        df['visit_datetime'] = pd.to_datetime(df['visit_datetime'])
        df['reserve_datetime'] = pd.to_datetime(df['reserve_datetime'])
        df['diff_vr'] = (df['visit_datetime'] - df['reserve_datetime']).dt.days
        df['visit_datetime'] = df['visit_datetime'].dt.date
        df['reserve_datetime'] = df['reserve_datetime'].dt.date
        # 每天的平均预订量和预定创建和执行之间的平均差
        grouped = df.groupby('visit_datetime',as_index = False)
        mean_lst.append(grouped.agg('mean'))
    
    for i in range(len(mean_lst)):
        mean_lst[i]['reserve_visitors'] = preprocessing.scale(mean_lst[i]['reserve_visitors'])
        mean_lst[i]['diff_vr'] = preprocessing.scale(mean_lst[i]['diff_vr'])

    res = pd.merge(mean_lst[0],mean_lst[1], on='visit_datetime',how='outer')
    res.rename(columns={'reserve_visitors_x':'rv1','reserve_visitors_y':'rv2',
                        'diff_vr_x':'diff1','diff_vr_y':'diff2',
                        'visit_datetime':'visit_date'}, inplace=True)
    hol.rename(columns={'calendar_date':'visit_date'}, inplace=True)
    hol['visit_date'] = pd.to_datetime(hol['visit_date'])
    hol['visit_date'] = hol['visit_date'].dt.date
    res = res.merge(hol, how='left',on='visit_date')
    return res


# fea4 : visitors
def visitors_of_stores(test, train):
    unique_stores = test['air_store_id'].unique()
    stores = pd.concat([DataFrame({'air_store_id': unique_stores, 
                                   'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)
    # 每周每家店的最小值
    tmp = train.groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
    stores = stores.merge(tmp,how='left', on=['air_store_id','dow'])
    # 每周每家店的最大值
    tmp = train.groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
    stores = stores.merge(tmp,how='left', on=['air_store_id','dow'])
    # 每周每家店的平均值
    tmp = train.groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
    stores = stores.merge(tmp,how='left', on=['air_store_id','dow'])
    # 每周每家店的中位数
    tmp = train.groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
    stores = stores.merge(tmp,how='left', on=['air_store_id','dow'])
    tmp = train.groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
    stores = stores.merge(tmp,how='left', on=['air_store_id','dow'])
    
    return stores

# fea5: store_info
def add_store_info(stores, asi):
    stores = stores.merge(asi, how='left',on='air_store_id')
    
    stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))
    stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))
    
    lbl = preprocessing.LabelEncoder()
    for i in range(7):
        stores['air_genre_name'+str(i)] = lbl.fit_transform(stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
        stores['air_area_name'+str(i)] = lbl.fit_transform(stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
    stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
    stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])
    
    stores['st_lat'] = stores['latitude'].max() - stores['latitude']
    stores['st_long'] = stores['longitude'].max() - stores['longitude']
    stores['lat_plus_long'] = stores['latitude'] + stores['longitude']
    stores['lat_plus_long'] = stores['lat_plus_long'].max() - stores['lat_plus_long']
    
    stores.drop(['air_genre_name3','air_genre_name4','air_genre_name5','air_genre_name6','latitude','longitude'],axis=1, inplace=True)
    return stores

def add_all(train, test, res,stores):
    train = train.merge(stores,how='left',on=['air_store_id','dow'])
    test = test.merge(stores,how='left',on=['air_store_id','dow'])
    train = train.merge(res,how='left',on='visit_date')
    test = test.merge(res,how='left',on='visit_date')
    return train, test

# fea6 store_id
def add_store_id(train,test):
    lbl = preprocessing.LabelEncoder()
    train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
    test['air_store_id2'] = lbl.transform(test['air_store_id'])

def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5

def considering_weight(res):
    # Considering Weights
    visit_data = pd.read_csv('../input/air_visit_data.csv',index_col= None)
    visit_data['visit_date'] = pd.to_datetime(visit_data['visit_date'])
    visit_data['dow'] = visit_data['visit_date'].dt.dayofweek
    visit_data['visit_date'] = visit_data['visit_date'].dt.date
    res['weights'] = ((res.index + 1) / len(res)) ** 5
    date_info = res[['visit_date','holiday_flg','weights','Consecutive_holidays']]
    visit_data = visit_data.merge(date_info, on='visit_date', how='left')
    visit_data['visitors'] = visit_data.visitors.map(pd.np.log1p)
    # create weights
    wmean = lambda x:( (x.weights * x.visitors).sum() / x.weights.sum() )
    
    visitors = visit_data.groupby(['air_store_id', 'dow', 'holiday_flg','Consecutive_holidays']).apply(wmean).reset_index()
    visitors.rename(columns={0:'visitors'}, inplace=True)
    
    sample_ = pd.read_csv('../input/sample_submission.csv', index_col=None)
    sample_['air_store_id'] = sample_.id.map(lambda x: '_'.join(x.split('_')[:-1]))
    sample_['visit_date'] = sample_.id.map(lambda x: x.split('_')[2])
    sample_.drop('visitors', axis=1, inplace=True)
    sample_['visit_date'] = pd.to_datetime(sample_['visit_date'])
    sample_['dow'] = sample_['visit_date'].dt.dayofweek
    sample_['visit_date'] = sample_['visit_date'].dt.date
    sample_ = sample_.merge(res[['visit_date','holiday_flg','Consecutive_holidays']], on='visit_date', how='left')
    sample_ = sample_.merge(visitors, on=[
    'air_store_id', 'dow', 'holiday_flg','Consecutive_holidays'], how='left')
    
    missings = sample_['visitors'].isnull()
    visitors = visit_data.groupby(['air_store_id', 'dow', 'holiday_flg']).apply(wmean).reset_index()
    visitors.rename(columns={0:'visitors'}, inplace=True)
    sample_.loc[missings, 'visitors'] = sample_[missings].merge(visitors, on=('air_store_id', 'dow', 'holiday_flg'), how='left')['visitors_y'].values
    
    missings = sample_['visitors'].isnull()
    visitors = visit_data.groupby(['air_store_id', 'dow']).apply(wmean).reset_index()
    visitors.rename(columns={0:'visitors'}, inplace=True)
    sample_.loc[missings, 'visitors'] = sample_[missings].merge(visitors, on=('air_store_id', 'dow'), how='left')['visitors_y'].values
    
    missings = sample_['visitors'].isnull()
    visitors = visit_data.groupby(['air_store_id']).apply(wmean).reset_index()
    visitors.rename(columns={0:'visitors'}, inplace=True)
    sample_.loc[missings, 'visitors'] = sample_[missings].merge(visitors, on=('air_store_id'), how='left')['visitors_y'].values
    return sample_

if __name__ == '__main__':
    #1 read_file
    ar = pd.read_csv('../input/air_reserve.csv',index_col= None)
    hr = pd.read_csv('../input/hpg_reserve.csv',index_col= None)
    train = pd.read_csv('../input/air_visit_data.csv',index_col= None)
    test = pd.read_csv('../input/sample_submission.csv', index_col=None)
    asi = pd.read_csv('../input/air_store_info.csv',index_col=None)
    hsi = pd.read_csv('../input/hpg_store_info.csv',index_col=None)
    hol = pd.read_csv('../input/date_info.csv', index_col=None)
    store_relation = pd.read_csv('../input/store_id_relation.csv',index_col=None)
    
    #2 fea3: year/month/day_of_week
    train['visit_date'] = pd.to_datetime(train['visit_date'])
    train['dow'] = train['visit_date'].dt.dayofweek
    train['year'] = train['visit_date'].dt.year
    train['month'] = train['visit_date'].dt.month
    train['visit_date'] = train['visit_date'].dt.date
    train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)

    test['visit_date'] = test['id'].map(lambda x: str(x).split('_')[2])
    test['air_store_id'] = test['id'].map(lambda x: '_'.join(x.split('_')[:2]))
    test['visit_date'] = pd.to_datetime(test['visit_date'])
    test['dow'] = test['visit_date'].dt.dayofweek
    test['year'] = test['visit_date'].dt.year
    test['month'] = test['visit_date'].dt.month
    test['visit_date'] = test['visit_date'].dt.date
    
    #3 fea1: Consecutive_hol
    get_consecutive_hol(hol)
    
    #4 fea2: diff_vr/rv_sum
    res = get_diff_vr_rv_sum([ar,hr], hol)
    
    #5 fea4: visitors
    stores = visitors_of_stores(test, train)
    
    #6 fea5: store_info
    stores = add_store_info(stores, asi)
    
    #7 fea6: store_id
    add_store_id(train, test)
    
    #8 add_all
    train, test = add_all(train, test, res, stores)
    
    col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date','visitors','day_of_week','air_area_name','air_genre_name','diff2','rv2','diff1','rv1']]
    train = train.fillna(-1)
    test = test.fillna(-1)
    
    # GBRT
    model1 = ensemble.GradientBoostingRegressor(learning_rate=0.15, n_estimators=300, max_depth=8, subsample=0.8)
    # XGBregressor
    model2 = XGBRegressor(learning_rate=0.15, n_estimators=600, subsample=0.8,colsample_bytree=0.8, max_depth=8, min_child_weight = 3, n_jobs=-1)
    
    # KNN
    model3 = neighbors.KNeighborsRegressor(n_neighbors=4, n_jobs=-1)
    
    # RF
    #model4 = ensemble.RandomForestRegressor(n_estimators=500,oob_score=True,n_jobs=-1,max_leaf_nodes=4)
    
    
    # Training!
    model1.fit(train[col], np.log1p(train['visitors'].values))
    model2.fit(train[col], np.log1p(train['visitors'].values))
    model3.fit(train[col], np.log1p(train['visitors'].values))
    #model4.fit(train[col], np.log1p(train['visitors'].values))
    
    preds1 = model1.predict(train[col])
    preds2 = model2.predict(train[col])
    preds3 = model3.predict(train[col])
    #preds4 = model4.predict(train[col])
    
    print('RMSE GradientBoostingRegressor: ', RMSLE(np.log1p(train['visitors'].values), preds1))
    print('RMSE XGBRegressor: ', RMSLE(np.log1p(train['visitors'].values), preds2))
    print('RMSE KNeighborsRegressor: ', RMSLE(np.log1p(train['visitors'].values), preds3))
    #print('RMSE RandomForestRegressor: ', RMSLE(np.log1p(train['visitors'].values), preds4))
    
    preds1 = model1.predict(test[col])
    preds2 = model2.predict(test[col])
    preds3 = model3.predict(test[col])
    #preds4 = model4.predict(test[col])
    
    sub2 = considering_weight(res)
    sub2['visitors'] = sub2.visitors.map(pd.np.expm1)
    test['visitors'] = 0.4*preds1 + 0.4*preds2 + 0.2*preds3
    test['visitors'] = np.expm1(test['visitors']).clip(lower=0.)
    sub1 = test[['id','visitors']].copy()
    
    sub_merge = pd.merge(sub1,sub2[['id','visitors']], on='id')
    sub_merge['visitors'] = sub_merge['visitors_x'] * 0.9 + sub_merge['visitors_y'] * 0.1
    sub_merge[['id','visitors']].to_csv('sub1.csv',index=False)