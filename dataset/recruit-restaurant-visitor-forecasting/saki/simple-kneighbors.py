import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import csv
from statistics import mean, median,variance,stdev
from datetime import datetime
import glob, re
from sklearn import *


air_reserve = pd.read_csv("../input/air_reserve.csv")
air_visit_data = pd.read_csv("../input/air_visit_data.csv")
air_store_info = pd.read_csv("../input/air_store_info.csv")
hpg_reserve = pd.read_csv("../input/hpg_reserve.csv")
hpg_store_info = pd.read_csv("../input/hpg_store_info.csv")
store_id_relation = pd.read_csv("../input/store_id_relation.csv")
date_info = pd.read_csv("../input/date_info.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")


hpg_reserve = pd.merge(hpg_reserve, store_id_relation, how='left', on=['hpg_store_id'])
air_reserve = pd.merge(air_reserve, store_id_relation, how='left', on=['air_store_id'])


hpg_reserve['visit_datetime'] = pd.to_datetime(hpg_reserve['visit_datetime'])
hpg_reserve['visit_year'] = hpg_reserve['visit_datetime'].dt.year
hpg_reserve['visit_month'] = hpg_reserve['visit_datetime'].dt.month
hpg_reserve['visit_date'] = hpg_reserve['visit_datetime'].dt.date
hpg_reserve = hpg_reserve.drop('visit_datetime',axis=1)

air_reserve['visit_datetime'] = pd.to_datetime(air_reserve['visit_datetime'])
air_reserve['visit_year'] = air_reserve['visit_datetime'].dt.year
air_reserve['visit_month'] = air_reserve['visit_datetime'].dt.month
air_reserve['visit_date'] = air_reserve['visit_datetime'].dt.date
air_reserve = air_reserve.drop('visit_datetime',axis=1)

hpg_reserve['reserve_datetime'] = pd.to_datetime(hpg_reserve['reserve_datetime'])
hpg_reserve['reserve_year'] = hpg_reserve['reserve_datetime'].dt.year
hpg_reserve['reserve_month'] = hpg_reserve['reserve_datetime'].dt.month
hpg_reserve['reserve_date'] = hpg_reserve['reserve_datetime'].dt.date
hpg_reserve = hpg_reserve.drop('reserve_datetime',axis=1)

air_reserve['reserve_datetime'] = pd.to_datetime(air_reserve['reserve_datetime'])
air_reserve['reserve_year'] = air_reserve['reserve_datetime'].dt.year
air_reserve['reserve_month'] = air_reserve['reserve_datetime'].dt.month
air_reserve['reserve_date'] = air_reserve['reserve_datetime'].dt.date
air_reserve = air_reserve.drop('reserve_datetime',axis=1)

air_visit_data['visit_datetime'] = pd.to_datetime(air_visit_data['visit_date'])
air_visit_data['visit_year'] = air_visit_data['visit_datetime'].dt.year
air_visit_data['visit_month'] = air_visit_data['visit_datetime'].dt.month
air_visit_data['visit_date'] = air_visit_data['visit_datetime'].dt.date
air_visit_data = air_visit_data.rename(columns={'visit_date':'visit_date'})
air_visit_data = air_visit_data.drop('visit_datetime',axis=1)

date_info['visit_day'] = date_info['calendar_date'].map(lambda x: (x.split('-')[2]))
date_info['mnd_flg'] = date_info['visit_day'].map(lambda x: 1 if int(x)>=25 else 0)
date_info['calendar_datetime'] = pd.to_datetime(date_info['calendar_date'])
date_info['visit_year'] = date_info['calendar_datetime'].dt.year
date_info['visit_month'] = date_info['calendar_datetime'].dt.month
date_info['calendar_date'] = date_info['calendar_datetime'].dt.date
date_info['long'] = (len(date_info)) - (date_info.index)
date_info['weight'] = ((date_info.index + 1) / len(date_info)) ** 5
date_info = date_info.rename(columns={'calendar_date':'visit_date'})
date_info = date_info.rename(columns={'calendar_datetime':'visit_datetime'})
non_bu = date_info.apply((lambda x:(x.day_of_week=='Sunday' or x.day_of_week=='Saturday'or x.day_of_week=='Friday') or x.holiday_flg==1), axis=1)
date_info = date_info.assign(non_buis_day = non_bu)
date_info = date_info.drop('visit_datetime',axis=1)
wkend_holidays = date_info.apply((lambda x:(x.day_of_week=='Sunday' or x.day_of_week=='Saturday') and x.holiday_flg==1), axis=1)
date_info.loc[wkend_holidays, 'holiday_flg'] = 0


print("===================================Success input data===================================")


sample_submission['air_store_id'] = sample_submission['id'].map(lambda x: '_'.join(x.split('_')[:2]))
sample_submission['visit_datetime'] = sample_submission['id'].map(lambda x: str(x).split('_')[2])
sample_submission['visit_datetime'] = pd.to_datetime(sample_submission['visit_datetime'])
sample_submission['visit_year'] = sample_submission['visit_datetime'].dt.year
sample_submission['visit_month'] = sample_submission['visit_datetime'].dt.month
sample_submission['visit_date'] = sample_submission['visit_datetime'].dt.date

sample_submission = pd.merge(sample_submission, date_info, how = 'left', on = ['visit_date','visit_year','visit_month'])
sample_submission = pd.merge(sample_submission, air_store_info, how = 'left', on = ['air_store_id'])
sample_submission = sample_submission.drop('visitors',axis=1)

print("sample_submission.isnull.sum")
print(sample_submission.isnull().sum())

sample_submission = sample_submission.fillna(0)
print("sample_submission1")
print(sample_submission)


air_visit_data = pd.merge(air_visit_data, date_info, how = 'left',on = ['visit_date','visit_year','visit_month'])
air_visit_data = pd.merge(air_visit_data, air_store_info, how = 'left', on = ['air_store_id'])

visit_data = air_visit_data[['air_store_id','day_of_week','holiday_flg','visitors','weight']]
wmean = lambda x:( (x.weight * x.visitors).sum() / x.weight.sum() )
df_ah_dh = visit_data.groupby(['air_store_id','day_of_week','holiday_flg']).apply(wmean).reset_index()
df_ah_dh.rename(columns={0:'visitors'}, inplace=True)
visit_data = air_visit_data[['air_store_id','non_buis_day','visitors','weight']]
df_ah_wh = visit_data.groupby(['air_store_id','non_buis_day']).apply(wmean).reset_index()
df_ah_wh.rename(columns={0:'visitors'}, inplace=True)

sample_submission2 = pd.merge(sample_submission,df_ah_dh, how='left', on=['air_store_id','holiday_flg','day_of_week'])
print("sample_submission2")
print(sample_submission2)
sample_visitors = sample_submission2.visitors
sample_ids = sample_submission2
print(sample_submission2.isnull().sum())

sample_submission2_nan = sample_submission2.visitors.isnull()
sample_submission2_null = sample_submission2[sample_submission2_nan]
sample_submission2_null = sample_submission2_null.drop('visitors',axis=1)


sub = sample_submission2_null

sample_test2 = sub.drop(['id','visit_date','visit_year','visit_datetime','visit_month','visit_day','latitude','longitude'], axis=1)

#sample_test2 = sample_submission.drop(['id','visit_date','visit_year','visit_datetime','visit_month','hpg_store_id','visit_day','latitude_x','longitude_x','hpg_area_name','latitude_y','longitude_y','hpg_genre_name'], axis=1)
sample_test2 = sample_test2.reset_index(drop=True)
sample_test2['air_genre_name'] = sample_test2['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))
sample_test2['air_area_name'] = sample_test2['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))
sample_test2['air_genre_name'] = sample_test2['air_genre_name'].map(lambda x: str(x).split(' ')[0])
sample_test2['air_area_name'] = sample_test2['air_area_name'].map(lambda x: str(x).split(' ')[0])

#sample_test = sub.drop(['id','visit_date','visit_datetime','visit_day','visit_year','visit_month','hpg_store_id','air_area_name','latitude_x','longitude_x','hpg_area_name','latitude_y','longitude_y','hpg_genre_name'], axis=1)
#sample_test = sample_test.reset_index(drop=True)

#print("sample_test")
#print(sample_test)

air_visit_test = air_visit_data
air_visit_test['air_genre_name'] = air_visit_test['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))
air_visit_test['air_area_name'] = air_visit_test['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))
air_visit_test['air_genre_name'] = air_visit_test['air_genre_name'].map(lambda x: str(x).split(' ')[0])
air_visit_test['air_area_name'] = air_visit_test['air_area_name'].map(lambda x: str(x).split(' ')[0])

#air_visit_test['sample_id'] = air_visit_test['air_store_id'].isin(sample_test['air_store_id'])
#air_visit_test['sample_genre'] = air_visit_test['air_genre_name'].isin(sample_test['air_genre_name'])
air_visit_test['sample_id'] = air_visit_test['air_store_id'].isin(sample_test2['air_store_id'])
air_visit_test['sample_genre'] = air_visit_test['air_genre_name'].isin(sample_test2['air_genre_name'])
air_visit_test['sample_area'] = air_visit_test['air_area_name'].isin(sample_test2['air_area_name'])
air_visit_test.loc[air_visit_test.sample_id == False,'air_store_id'] = 0
air_visit_test.loc[air_visit_test.sample_genre == False,'air_genre_name'] = 0
air_visit_test.loc[air_visit_test.sample_area == False,'air_area_name'] = 0
air_visit_test = air_visit_test.drop(['sample_genre','sample_area','sample_id','visit_year','visit_month','visit_date','visit_day','latitude','longitude'], axis=1)

print("air_visit_test")
print(air_visit_test)


#sample_test = pd.get_dummies(sample_test, columns = ['day_of_week'])
#sample_test = pd.get_dummies(sample_test, columns = ['air_genre_name'])
#sample_test = pd.get_dummies(sample_test, columns = ['air_store_id'])
sample_test2 = pd.get_dummies(sample_test2, columns = ['day_of_week'])
sample_test2 = pd.get_dummies(sample_test2, columns = ['air_genre_name'])
sample_test2 = pd.get_dummies(sample_test2, columns = ['air_store_id'])
sample_test2 = pd.get_dummies(sample_test2, columns = ['air_area_name'])
air_visit_test = pd.get_dummies(air_visit_test, columns = ['day_of_week'])
air_visit_test = pd.get_dummies(air_visit_test, columns = ['air_genre_name'])
air_visit_test = pd.get_dummies(air_visit_test, columns = ['air_store_id'])
air_visit_test = pd.get_dummies(air_visit_test, columns = ['air_area_name'])
air_visit_test = air_visit_test.drop('air_store_id_0',axis=1)
#air_visit_test = air_visit_test.drop('visit_month_0',axis=1)
air_visit_test = air_visit_test.drop('air_genre_name_0',axis=1)


#sample_test.non_buis_day = sample_test.non_buis_day.astype('int')
sample_test2.non_buis_day = sample_test2.non_buis_day.astype('int')
air_visit_test.non_buis_day = air_visit_test.non_buis_day.astype('int')

#print("sample_test columns")
#print(sample_test.columns)

print("sample_test2 columns")
print(sample_test2.columns)

print("air_visit_test columns")
print(air_visit_test.columns)


xs = air_visit_test.drop('visitors',axis=1)
y = air_visit_test.visitors

forest = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=4)
forest.fit(xs,y)

#visitors_pre = forest.predict(sample_test)
visitors_pre = forest.predict(sample_test2)
visitors_pre = pd.Series(visitors_pre)

print(visitors_pre)

sample_visitors = sample_visitors.dropna()
sample_visitors = sample_visitors.append(visitors_pre,ignore_index=True)
#sample_visitors = visitors_pre
print("sample_visitors")
print(sample_visitors)
sample_ids = sample_ids.dropna()
sample_ids = sample_ids.id
sample_ids = sample_ids.append(sample_submission2_null.id,ignore_index=True)
#sample_ids = sample_submission.id
print(sample_visitors)
print(sample_ids)

submit = pd.concat([sample_ids, sample_visitors], axis=1,ignore_index=True)
submit.columns = ['id','visitors']

print(submit)


submit.to_csv('submit.csv', index=False)