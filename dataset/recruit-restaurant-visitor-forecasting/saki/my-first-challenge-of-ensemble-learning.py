import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import csv
from statistics import mean, median,variance,stdev
from datetime import datetime
import glob, re
from sklearn import *
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor



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
print(date_info)

print("===================================Success input data===================================")


sample_submission['air_store_id'] = sample_submission['id'].map(lambda x: '_'.join(x.split('_')[:2]))
sample_submission['visit_datetime'] = sample_submission['id'].map(lambda x: str(x).split('_')[2])
sample_submission['visit_datetime'] = pd.to_datetime(sample_submission['visit_datetime'])
sample_submission['visit_year'] = sample_submission['visit_datetime'].dt.year
sample_submission['visit_month'] = sample_submission['visit_datetime'].dt.month
sample_submission['visit_date'] = sample_submission['visit_datetime'].dt.date

sample_submission = pd.merge(sample_submission, date_info, how = 'left', on = ['visit_date','visit_year','visit_month'])
sample_submission = pd.merge(sample_submission, store_id_relation, how = 'left', on = ['air_store_id'])
sample_submission = pd.merge(sample_submission, air_store_info, how = 'left', on = ['air_store_id'])
sample_submission = pd.merge(sample_submission, hpg_store_info, how = 'left', on = ['hpg_store_id'])
sample_submission = sample_submission.drop('visitors',axis=1)

sample_submission = sample_submission.fillna(0)
print("sample_submission1")
print(sample_submission)


air_visit_data = pd.merge(air_visit_data, date_info, how = 'left',on = ['visit_date','visit_year','visit_month'])
air_visit_data = pd.merge(air_visit_data, store_id_relation, how = 'left', on = ['air_store_id'])
air_visit_data = pd.merge(air_visit_data, air_store_info, how = 'left', on = ['air_store_id'])
air_visit_data = pd.merge(air_visit_data, hpg_store_info, how = 'left', on = ['hpg_store_id'])

visit_data = air_visit_data[['air_store_id','day_of_week','holiday_flg','visitors','weight']]
wmean = lambda x:( (x.weight * x.visitors).sum() / x.weight.sum() )
df_ah_dh = visit_data.groupby(['air_store_id','day_of_week','holiday_flg']).apply(wmean).reset_index()
df_ah_dh.rename(columns={0:'visitors'}, inplace=True)
visit_data = air_visit_data[['air_store_id','non_buis_day','visitors','weight']]
df_ah_wh = visit_data.groupby(['air_store_id','non_buis_day']).apply(wmean).reset_index()
df_ah_wh.rename(columns={0:'visitors'}, inplace=True)

sample_submission2 = pd.merge(sample_submission,df_ah_dh, how='left', on=['air_store_id','day_of_week','holiday_flg'])
print("sample_submission2")
print(sample_submission2)
print(sample_submission2.isnull().sum())

sample_submission2_nan = sample_submission2.visitors.isnull()
sample_submission2_null = sample_submission2[sample_submission2_nan]
sample_submission2_null = sample_submission2_null.drop('visitors',axis=1)

sample_submission3 = pd.merge(sample_submission2_null,df_ah_wh, how='left', on=['air_store_id','non_buis_day'])
print("sample_submission3")
print(sample_submission3)
print(sample_submission3.isnull().sum())

sample_submission2 = sample_submission2.dropna()
sample_submission3 = sample_submission3.dropna()


subm = pd.concat([sample_submission2,sample_submission3],ignore_index = True)

submit1 = pd.concat([subm.id,subm.visitors],axis=1)
submit1.columns = ['id','visitors']
print(submit1)

sample_test2 = sample_submission

test_make = air_visit_data.groupby(['air_store_id','day_of_week'], as_index=False)['visitors'].median().reset_index().rename(columns={'visitors':'median_visitors'})
test_make = air_visit_data.groupby(['air_store_id','day_of_week'], as_index=False)['visitors'].mean().reset_index().rename(columns={'visitors':'mean_visitors'})
test_make = air_visit_data.groupby(['air_store_id','day_of_week'], as_index=False)['visitors'].sum().reset_index().rename(columns={'visitors':'sum_visitors'})
test_make = air_visit_data.groupby(['air_store_id','day_of_week'], as_index=False)['visitors'].max().reset_index().rename(columns={'visitors':'max_visitors'})
test_make = air_visit_data.groupby(['air_store_id','day_of_week'], as_index=False)['visitors'].min().reset_index().rename(columns={'visitors':'min_visitors'})
test_make = air_visit_data.groupby(['air_store_id','day_of_week'], as_index=False)['visitors'].count().reset_index().rename(columns={'visitors':'count_visitors'})

sample_test2 = pd.merge(sample_test2,test_make, how='left', on=['air_store_id','day_of_week'])
sample_test2['air_genre_name'] = sample_test2['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))
sample_test2['air_area_name'] = sample_test2['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))
sample_test2['air_genre_name'] = sample_test2['air_genre_name'].map(lambda x: str(x).split(' ')[0])
sample_test2['air_area_name'] = sample_test2['air_area_name'].map(lambda x: str(x).split(' ')[0])

#sample_test2 = sample_test2['median_visitors'].fillna(air_visit_data.groupby(['air_store_id','non_buis_day'], as_index=False)['visitors'].median())
#sample_test2 = sample_test2['mean_visitors'].fillna(air_visit_data.groupby(['air_store_id','non_buis_day'], as_index=False)['visitors'].mean())
#sample_test2 = sample_test2['sum_visitors'].fillna(air_visit_data.groupby(['air_store_id','non_buis_day'], as_index=False)['visitors'].sum())
#sample_test2 = sample_test2['max_visitors'].fillna(air_visit_data.groupby(['air_store_id','non_buis_day'], as_index=False)['visitors'].max())
#sample_test2 = sample_test2['min_visitors'].fillna(air_visit_data.groupby(['air_store_id','non_buis_day'], as_index=False)['visitors'].min())
sample_test2 = sample_test2.fillna(method='ffill')

sample_test2 = sample_test2.drop(['id','visit_date','visit_year','visit_datetime','visit_month','hpg_store_id','visit_day','latitude_x','longitude_x','hpg_area_name','latitude_y','longitude_y','hpg_genre_name'], axis=1)
sample_test2 = sample_test2.reset_index(drop=True)

air_visit_test = air_visit_data
air_visit_test = pd.merge(air_visit_test,test_make, how='left', on=['air_store_id','day_of_week'])
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
air_visit_test = air_visit_test.drop(['sample_genre','sample_area','sample_id','visit_year','visit_month','visit_date','visit_day','hpg_store_id','latitude_x','longitude_x','hpg_area_name','latitude_y','longitude_y','hpg_genre_name'], axis=1)
print("air_visit_test")
print(air_visit_test)

sample_test2 = pd.get_dummies(sample_test2, columns = ['day_of_week'])
sample_test2 = pd.get_dummies(sample_test2, columns = ['air_genre_name'])
sample_test2 = pd.get_dummies(sample_test2, columns = ['air_store_id'])
sample_test2 = pd.get_dummies(sample_test2, columns = ['air_area_name'])
air_visit_test = pd.get_dummies(air_visit_test, columns = ['day_of_week'])
air_visit_test = pd.get_dummies(air_visit_test, columns = ['air_genre_name'])
air_visit_test = pd.get_dummies(air_visit_test, columns = ['air_store_id'])
air_visit_test = pd.get_dummies(air_visit_test, columns = ['air_area_name'])
air_visit_test = air_visit_test.drop('air_store_id_0',axis=1)
#air_visit_test = air_visit_test.drop('air_genre_name_0',axis=1)
#air_visit_test = air_visit_test.drop('air_area_name_0',axis=1)

sample_test2.non_buis_day = sample_test2.non_buis_day.astype('int')
air_visit_test.non_buis_day = air_visit_test.non_buis_day.astype('int')

print("sample_test2 columns")
print(sample_test2.columns)

print("air_visit_test columns")
print(air_visit_test.columns)

xs = air_visit_test.drop('visitors',axis=1)
y = air_visit_test.visitors

model = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=4)
model.fit(xs.values,y.values)

#visitors_pre = forest.predict(sample_test)
visitors_pre1 = model.predict(sample_test2.values)
print("visitors_pre1")
print(visitors_pre1)

model2 = RandomForestRegressor()
model2.fit(xs.values,y.values)

visitors_pre2 = model2.predict(sample_test2.values)
print("visitors_pre2")
print(visitors_pre2)

#model3 = XGBRegressor(learning_rate=0.2, random_state=3, n_estimators=200, subsample=0.8, colsample_bytree=0.8, max_depth =10)
#model3.fit(xs.values,y.values)

#visitors_pre3 = model3.predict(sample_test2.values)
#print("visitors_pre3")
#print(visitors_pre3)

#model4 = ensemble.GradientBoostingRegressor()
#model4.fit(xs.values,y.values)
#visitors_pre4 = model4.predict(sample_test2.values)
#print("visitors_pre4")
#print(visitors_pre4)

#visitors_pre = 0.2*visitors_pre1+0.2*visitors_pre2+0.3*visitors_pre3+0.2*visitors_pre4
visitors_pre = (visitors_pre1+visitors_pre2)/2

visitors_pre1 = pd.Series(visitors_pre1)
sample_visitors1 = visitors_pre1
sample_ids1 = sample_submission.id
print(sample_visitors1)
print(sample_ids1)
submit1 = pd.concat([sample_ids1, sample_visitors1], axis=1,ignore_index=True)
submit1.columns = ['id','visitors']
submit1.to_csv('submit1.csv', index=False)

visitors_pre2 = pd.Series(visitors_pre2)
sample_visitors2 = visitors_pre2
sample_ids2 = sample_submission.id
print(sample_visitors2)
print(sample_ids2)
submit2 = pd.concat([sample_ids2, sample_visitors2], axis=1,ignore_index=True)
submit2.columns = ['id','visitors']
submit2.to_csv('submit2.csv', index=False)

#visitors_pre3 = pd.Series(visitors_pre3)
#sample_visitors3 = visitors_pre3
#sample_ids3 = sample_submission.id
#print(sample_visitors3)
#print(sample_ids3)
#submit3 = pd.concat([sample_ids3, sample_visitors3], axis=1,ignore_index=True)
#submit3.columns = ['id','visitors']
#submit3.to_csv('submit3.csv', index=False)

#visitors_pre4 = pd.Series(visitors_pre4)
#sample_visitors4 = visitors_pre4
#sample_ids4 = sample_submission.id
#print(sample_visitors4)
#print(sample_ids4)
#submit4 = pd.concat([sample_ids4, sample_visitors4], axis=1,ignore_index=True)
#submit4.columns = ['id','visitors']
#submit4.to_csv('submit4.csv', index=False)

visitors_pre0 = pd.Series(visitors_pre)
sample_visitors0 = visitors_pre0
sample_ids0 = sample_submission.id
print(sample_visitors0)
print(sample_ids0)
submit0 = pd.concat([sample_ids0, sample_visitors0], axis=1,ignore_index=True)
submit0.columns = ['id','visitors']
submit0.to_csv('submit3.csv', index=False)

visitors_pre = 0.7*submit1['visitors'].values+0.3*visitors_pre
visitors_pre = pd.Series(visitors_pre)
print(visitors_pre)

sample_visitors = visitors_pre
sample_ids = sample_submission.id
print(sample_visitors)
print(sample_ids)

submit = pd.concat([sample_ids, sample_visitors], axis=1,ignore_index=True)
submit.columns = ['id','visitors']

print(submit)

#submit = submit1
#submit['visitors'] = 0.7*

submit.to_csv('submit.csv', index=False)