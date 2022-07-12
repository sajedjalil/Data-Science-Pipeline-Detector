# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 21:02:37 2018

@author: ZHILANGTAOSHA
"""
#来源于  https://www.kaggle.com/emorej/unexpected-10th-place-model
#英语不怎么好，所以请见谅 
#English is not good, please forgive me


import pandas as pd
import time
import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import date, timedelta
from sklearn.preprocessing import LabelEncoder
import datetime
train_df = pd.read_csv('../input/air_visit_data.csv', parse_dates=["visit_date"])
sample_df = pd.read_csv('../input/sample_submission.csv', quotechar="'")

train_df['dow'] = train_df['visit_date'].dt.dayofweek
train_df['day'] = train_df['visit_date'].dt.day
train_df['year'] = train_df['visit_date'].dt.year
train_df['month'] = train_df['visit_date'].dt.month
train_df['week'] = train_df['visit_date'].dt.week
train_df['visit_date'] = train_df['visit_date'].dt.date

sample_df['visit_date'] = sample_df['id'].map(lambda x: str(x).split('_')[2])
sample_df['air_store_id'] = sample_df['id'].map(lambda x: '_'.join(x.split('_')[:2]))
sample_df['visit_date'] = pd.to_datetime(sample_df['visit_date'])
sample_df['dow'] = sample_df['visit_date'].dt.dayofweek
sample_df['day'] = sample_df['visit_date'].dt.day
sample_df['year'] = sample_df['visit_date'].dt.year
sample_df['month'] = sample_df['visit_date'].dt.month
sample_df['week'] = sample_df['visit_date'].dt.week
sample_df['visit_date'] = sample_df['visit_date'].dt.date

################################################# Calendar processing
# Flags creation for the day before and the day after a holiday
# Creating an eighth day-of-week for holidays
# Flags creation for 7 days before Golden week, and 7 days after it

holiday_df = pd.read_csv('../input/date_info.csv', parse_dates=["calendar_date"])
holiday_df['calendar_date'] = holiday_df['calendar_date'].dt.date
holiday_df.columns = ['visit_date', 'dow', 'holiday_flg']

# day before and day after a holiday
holiday_df['visit_date_p1'] = holiday_df['visit_date'] + datetime.timedelta(days=1)
holiday_df['visit_date_m1'] = holiday_df['visit_date'] + datetime.timedelta(days=-1)

train_df = pd.merge(train_df, holiday_df[['visit_date', 'holiday_flg']], on='visit_date', how='left')
sample_df = pd.merge(sample_df, holiday_df[['visit_date', 'holiday_flg']], on='visit_date', how='left')

holiday_df.columns = ['ex_visit_date', 'dow', 'after_holiday_flg', 'visit_date', 'visit_date_m1']

train_df = pd.merge(train_df, holiday_df[['visit_date', 'after_holiday_flg']], on='visit_date', how='left')
sample_df = pd.merge(sample_df, holiday_df[['visit_date', 'after_holiday_flg']], on='visit_date', how='left')

holiday_df.columns = ['ex_visit_date', 'dow', 'before_holiday_flg', 'visit_date_p1', 'visit_date']

train_df = pd.merge(train_df, holiday_df[['visit_date', 'before_holiday_flg']], on='visit_date', how='left')
sample_df = pd.merge(sample_df, holiday_df[['visit_date', 'before_holiday_flg']], on='visit_date', how='left')

##### creating an eighth day a week :)
train_df.loc[train_df.holiday_flg==1,'dow'] = -1
sample_df.loc[sample_df.holiday_flg==1,'dow'] = -1

# week before and after Golden week
train_df['before_golden_flag'] = 0
train_df.loc[(train_df.visit_date<=datetime.datetime.strptime('2016-04-28', '%Y-%m-%d').date()) & (train_df.visit_date>=datetime.datetime.strptime('2016-04-22', '%Y-%m-%d').date()),'before_golden_flag'] = 1
train_df['after_golden_flag'] = 0
train_df.loc[(train_df.visit_date<=datetime.datetime.strptime('2016-05-15', '%Y-%m-%d').date()) & (train_df.visit_date>=datetime.datetime.strptime('2016-05-09', '%Y-%m-%d').date()),'after_golden_flag'] = 1
sample_df['before_golden_flag'] = 0
sample_df.loc[(sample_df.visit_date<=datetime.datetime.strptime('2017-04-28', '%Y-%m-%d').date()) & (sample_df.visit_date>=datetime.datetime.strptime('2017-04-22', '%Y-%m-%d').date()),'before_golden_flag'] = 1
sample_df['after_golden_flag'] = 0
sample_df.loc[(sample_df.visit_date<=datetime.datetime.strptime('2017-05-14', '%Y-%m-%d').date()) & (sample_df.visit_date>=datetime.datetime.strptime('2017-05-08', '%Y-%m-%d').date()),'after_golden_flag'] = 1
 
 
 
#################################### Restaurants informations processing

air_store_info_df = pd.read_csv('../input/rrv-weather-data/air_store_info_with_nearest_active_station.csv', usecols=[0,1,2,3,4,7])
hpg_store_info_df = pd.read_csv('../input/rrv-weather-data/hpg_store_info_with_nearest_active_station.csv', usecols=[0,1,2,3,4,7])

# split area name
air_store_info_df[['prefecture','city','district']] = pd.DataFrame([x.split(' ',2) for x in air_store_info_df['air_area_name'].tolist()])
hpg_store_info_df[['prefecture','city','district']] = pd.DataFrame([x.split(' ',2) for x in hpg_store_info_df['hpg_area_name'].tolist()])

# estimate number of restaurants per city, city/genre
air_store_info_df['store_id'] = air_store_info_df['air_store_id']
hpg_store_info_df['store_id'] = hpg_store_info_df['hpg_store_id']
air_store_info_df['genre_name'] = air_store_info_df['air_genre_name']
hpg_store_info_df['genre_name'] = hpg_store_info_df['hpg_genre_name']

store_info_df = pd.concat([air_store_info_df[['store_id','city','genre_name']], hpg_store_info_df[['store_id','city','genre_name']]]).reset_index(drop=True)

genres = {
    'Japanese style':'Japanese food',
    'Italian':'Italian/French',
    'International cuisine':'International cuisine',
    'Grilled meat':'Okonomiyaki/Monja/Teppanyaki',
    'Creation':'Creative cuisine',
    'Seafood':'Other',
    'Spain Bar/Italian Bar':'Italian/French',
    'Japanese food in general':'Japanese food',
    'Shabu-shabu/Sukiyaki':'Japanese food',
    'Chinese general':'Asian',
    'Creative Japanese food':'Japanese food',
    'Japanese cuisine/Kaiseki':'Japanese food',
    'Korean cuisine':'Yakiniku/Korean food',
    'Okonomiyaki/Monja/Teppanyaki':'Okonomiyaki/Monja/Teppanyaki',
    'Karaoke':'Karaoke/Party',
    'Steak/Hamburger/Curry':'Western food',
    'French':'Italian/French',
    'Cafe':'Cafe/Sweets',
    'Bistro':'Izakaya',
    'Sushi':'Japanese food',
    'Party':'Karaoke/Party',
    'Western food':'Western food',
    'Pasta/Pizza':'Italian/French',
    'Thai/Vietnamese food':'Asian',
    'Bar/Cocktail':'Bar/Cocktail',
    'Amusement bar':'Bar/Cocktail',
    'Cantonese food':'Asian',
    'Dim Sum/Dumplings':'Asian',
    'Sichuan food':'Asian',
    'Sweets':'Cafe/Sweets',
    'Spain/Mediterranean cuisine':'International cuisine',
    'Udon/Soba':'Japanese food',
    'Shanghai food':'Asian',
    'Taiwanese/Hong Kong cuisine':'Asian',
    'Japanese food':'Japanese food', 
    'Dining bar':'Dining bar', 
    'Izakaya':'Japanese food',
    'Italian/French':'Italian/French', 
    'Cafe/Sweets':'Cafe/Sweets',
    'Yakiniku/Korean food':'Yakiniku/Korean food', 
    'Western food':'Western food', 
    'Bar/Cocktail':'Bar/Cocktail', 
    'Other':'Other',
    'Creative cuisine':'Creative cuisine', 
    'Karaoke/Party':'Karaoke/Party', 
    'International cuisine':'International cuisine',
    'Asian':'Asian',
    'None':'None',
    'No Data':'No Data'}
store_info_df['genre_name'] = store_info_df['genre_name'].map(genres)


city_nbr_df = store_info_df.groupby('city')['store_id'].nunique().reset_index()
city_nbr_df.columns = ['city', 'stores_per_city']
citygenre_nbr_df = store_info_df.groupby(['city','genre_name'])['store_id'].nunique().reset_index()
citygenre_nbr_df.columns = ['city', 'genre_name', 'stores_per_citygenre']

air_store_info_df = pd.merge(air_store_info_df, city_nbr_df, on='city', how='left')
air_store_info_df = pd.merge(air_store_info_df, citygenre_nbr_df, on=['city', 'genre_name'], how='left')

# encoding
lbl = LabelEncoder()
air_store_info_df['air_genre_name'] = lbl.fit_transform(air_store_info_df['air_genre_name'])
air_store_info_df['prefecture'] = lbl.fit_transform(air_store_info_df['prefecture'])
air_store_info_df['city'] = lbl.fit_transform(air_store_info_df['city'])
air_store_info_df['district'] = lbl.fit_transform(air_store_info_df['district'])
air_store_info_df.drop(['air_area_name','store_id','genre_name'], axis=1, inplace=True)

train_df = pd.merge(train_df, air_store_info_df, on='air_store_id', how='left')
sample_df = pd.merge(sample_df, air_store_info_df, on='air_store_id', how='left')


######################################## Outliers

# flag outliers
out_flag = 1

def find_outliers(series):
    return (series - series.mean()) > 1.96 * series.std()

train_df['is_outlier'] = train_df.groupby(['air_store_id','dow']).apply(lambda x: find_outliers(x['visitors'])).values

if out_flag:
    train_df = train_df.loc[train_df.is_outlier == False]
    
########################################## Visitors stats

# rolling stats on visitors
# exponential weighted 21 days
# standard rolling 90 & 180 days
# groupby air_store, air_store/dow, district/genre, genre/holidayflag

stats_visit_df = train_df[['air_store_id','visit_date','visitors','dow','holiday_flg','air_genre_name','district']].copy()
stats_visit_df = pd.concat([stats_visit_df, sample_df[['air_store_id','visit_date','visitors','dow','holiday_flg','air_genre_name','district']]]).reset_index(drop=True)

# not found more elegant way to build a datetime index rolling properly... 
stats_visit_df.sort_values('visit_date', inplace=True)
stats_visit_df['increment'] = pd.timedelta_range('1 second',freq='1ns',periods=len(stats_visit_df.index))
stats_visit_df['dtindex'] = pd.to_datetime(stats_visit_df['visit_date']) + stats_visit_df['increment']
stats_visit_df = stats_visit_df.set_index('dtindex',drop=True)


fsum = lambda x: x.rolling(window="90D",closed='left').sum()
stats_visit_df['visitdow_m3_sum'] = stats_visit_df.groupby(['air_store_id','dow']).visitors.apply(fsum).fillna(0)
stats_visit_df['visit_m3_sum'] = stats_visit_df.groupby(['air_store_id']).visitors.apply(fsum).fillna(0)
stats_visit_df['districtgenre_m3_sum'] = stats_visit_df.groupby(['district','air_genre_name']).visitors.apply(fsum).fillna(0)

fsum = lambda x: x.rolling(window="180D",closed='left').sum()
stats_visit_df['visitdow_m6_sum'] = stats_visit_df.groupby(['air_store_id','dow']).visitors.apply(fsum).fillna(0)
stats_visit_df['visit_m6_sum'] = stats_visit_df.groupby(['air_store_id']).visitors.apply(fsum).fillna(0)
stats_visit_df['districtgenre_m6_sum'] = stats_visit_df.groupby(['district','air_genre_name']).visitors.apply(fsum).fillna(0)
stats_visit_df['genrehol_m6_sum'] = stats_visit_df.groupby(['air_genre_name','holiday_flg']).visitors.apply(fsum).fillna(0)

stats_visit_df['visitors'] = stats_visit_df['visitors'].replace(0, np.NaN)


fmean_ewm21 = lambda x: x.shift().ewm(span=21,adjust=True,ignore_na=True,min_periods=3).mean()
fstd_ewm21 = lambda x: x.shift().ewm(span=21,adjust=True,ignore_na=True,min_periods=3).std()
stats_visit_df['visitdow_ewm21_mean'] = stats_visit_df.groupby(['air_store_id','dow']).visitors.apply(fmean_ewm21).fillna(0)
stats_visit_df['visitdow_ewm21_std'] = stats_visit_df.groupby(['air_store_id','dow']).visitors.apply(fstd_ewm21).fillna(0)
stats_visit_df['visit_ewm21_mean'] = stats_visit_df.groupby(['air_store_id']).visitors.apply(fmean_ewm21).fillna(0)
stats_visit_df['visit_ewm21_std'] = stats_visit_df.groupby(['air_store_id']).visitors.apply(fstd_ewm21).fillna(0)
stats_visit_df['districtgenre_ewm21_mean'] = stats_visit_df.groupby(['district','air_genre_name']).visitors.apply(fmean_ewm21).fillna(0)
stats_visit_df['districtgenre_ewm21_std'] = stats_visit_df.groupby(['district','air_genre_name']).visitors.apply(fstd_ewm21).fillna(0)


fmean = lambda x: x.rolling(window="90D",min_periods=3,closed='left').mean()
fstd = lambda x: x.rolling(window="90D",min_periods=3,closed='left').std()
fcount = lambda x: x.rolling(window="90D",closed='left').count()
fmin = lambda x: x.rolling(window="90D",min_periods=3,closed='left').min()
fmax = lambda x: x.rolling(window="90D",min_periods=3,closed='left').max()
stats_visit_df['visitdow_m3_mean'] = stats_visit_df.groupby(['air_store_id','dow']).visitors.apply(fmean).fillna(0)
stats_visit_df['visitdow_m3_std'] = stats_visit_df.groupby(['air_store_id','dow']).visitors.apply(fstd).fillna(0)
stats_visit_df['visitdow_m3_count'] = stats_visit_df.groupby(['air_store_id','dow']).visitors.apply(fcount).fillna(0)
stats_visit_df['visitdow_m3_min'] = stats_visit_df.groupby(['air_store_id','dow']).visitors.apply(fmin).fillna(0)
stats_visit_df['visitdow_m3_max'] = stats_visit_df.groupby(['air_store_id','dow']).visitors.apply(fmax).fillna(0)
stats_visit_df['visit_m3_mean'] = stats_visit_df.groupby(['air_store_id']).visitors.apply(fmean).fillna(0)
stats_visit_df['visit_m3_std'] = stats_visit_df.groupby(['air_store_id']).visitors.apply(fstd).fillna(0)
stats_visit_df['visit_m3_count'] = stats_visit_df.groupby(['air_store_id']).visitors.apply(fcount).fillna(0)
stats_visit_df['visit_m3_min'] = stats_visit_df.groupby(['air_store_id']).visitors.apply(fmin).fillna(0)
stats_visit_df['visit_m3_max'] = stats_visit_df.groupby(['air_store_id']).visitors.apply(fmax).fillna(0)
stats_visit_df['districtgenre_m3_mean'] = stats_visit_df.groupby(['district','air_genre_name']).visitors.apply(fmean).fillna(0)
stats_visit_df['districtgenre_m3_std'] = stats_visit_df.groupby(['district','air_genre_name']).visitors.apply(fstd).fillna(0)
stats_visit_df['districtgenre_m3_count'] = stats_visit_df.groupby(['district','air_genre_name']).visitors.apply(fcount).fillna(0)
stats_visit_df['districtgenre_m3_min'] = stats_visit_df.groupby(['district','air_genre_name']).visitors.apply(fmin).fillna(0)
stats_visit_df['districtgenre_m3_max'] = stats_visit_df.groupby(['district','air_genre_name']).visitors.apply(fmax).fillna(0)


fmean = lambda x: x.rolling(window="180D",min_periods=3,closed='left').mean()
fstd = lambda x: x.rolling(window="180D",min_periods=3,closed='left').std()
fcount = lambda x: x.rolling(window="180D",closed='left').count()
fmin = lambda x: x.rolling(window="180D",min_periods=3,closed='left').min()
fmax = lambda x: x.rolling(window="180D",min_periods=3,closed='left').max()
stats_visit_df['visitdow_m6_mean'] = stats_visit_df.groupby(['air_store_id','dow']).visitors.apply(fmean).fillna(0)
stats_visit_df['visitdow_m6_std'] = stats_visit_df.groupby(['air_store_id','dow']).visitors.apply(fstd).fillna(0)
stats_visit_df['visitdow_m6_count'] = stats_visit_df.groupby(['air_store_id','dow']).visitors.apply(fcount).fillna(0)
stats_visit_df['visitdow_m6_min'] = stats_visit_df.groupby(['air_store_id','dow']).visitors.apply(fmin).fillna(0)
stats_visit_df['visitdow_m6_max'] = stats_visit_df.groupby(['air_store_id','dow']).visitors.apply(fmax).fillna(0)
stats_visit_df['visit_m6_mean'] = stats_visit_df.groupby(['air_store_id']).visitors.apply(fmean).fillna(0)
stats_visit_df['visit_m6_std'] = stats_visit_df.groupby(['air_store_id']).visitors.apply(fstd).fillna(0)
stats_visit_df['visit_m6_count'] = stats_visit_df.groupby(['air_store_id']).visitors.apply(fcount).fillna(0)
stats_visit_df['visit_m6_min'] = stats_visit_df.groupby(['air_store_id']).visitors.apply(fmin).fillna(0)
stats_visit_df['visit_m6_max'] = stats_visit_df.groupby(['air_store_id']).visitors.apply(fmax).fillna(0)
stats_visit_df['districtgenre_m6_mean'] = stats_visit_df.groupby(['district','air_genre_name']).visitors.apply(fmean).fillna(0)
stats_visit_df['districtgenre_m6_std'] = stats_visit_df.groupby(['district','air_genre_name']).visitors.apply(fstd).fillna(0)
stats_visit_df['districtgenre_m6_count'] = stats_visit_df.groupby(['district','air_genre_name']).visitors.apply(fcount).fillna(0)
stats_visit_df['districtgenre_m6_min'] = stats_visit_df.groupby(['district','air_genre_name']).visitors.apply(fmin).fillna(0)
stats_visit_df['districtgenre_m6_max'] = stats_visit_df.groupby(['district','air_genre_name']).visitors.apply(fmax).fillna(0)
stats_visit_df['genrehol_m6_mean'] = stats_visit_df.groupby(['air_genre_name','holiday_flg']).visitors.apply(fmean).fillna(0)
stats_visit_df['genrehol_m6_std'] = stats_visit_df.groupby(['air_genre_name','holiday_flg']).visitors.apply(fstd).fillna(0)
stats_visit_df['genrehol_m6_count'] = stats_visit_df.groupby(['air_genre_name','holiday_flg']).visitors.apply(fcount).fillna(0)
stats_visit_df['genrehol_m6_min'] = stats_visit_df.groupby(['air_genre_name','holiday_flg']).visitors.apply(fmin).fillna(0)
stats_visit_df['genrehol_m6_max'] = stats_visit_df.groupby(['air_genre_name','holiday_flg']).visitors.apply(fmax).fillna(0)

# nb of visitors 52 weeks ago, and 51-53 weeks ago
fsum = lambda x: x.rolling(window="365D").sum() - x.rolling(window="358D").sum()

stats_visit_df['visit_lasty'] = stats_visit_df.groupby(['air_store_id','dow']).visitors.apply(fsum).fillna(0)

fsum = lambda x: x.rolling(window="372D").sum() - x.rolling(window="351D").sum()
stats_visit_df['visit_lasty2'] = stats_visit_df.groupby(['air_store_id','dow']).visitors.apply(fsum).fillna(0)


stats_visit_df.drop(['dow','visitors','increment','holiday_flg','air_genre_name','district'], axis=1, inplace=True)

train_df = pd.merge(train_df, stats_visit_df, on=['air_store_id','visit_date'], how='left')
sample_df = pd.merge(sample_df, stats_visit_df, on=['air_store_id','visit_date'], how='left')

print('处理天气')
############################################## Weather informations: precipitation, avg_temperature & hours_sunlight


st = {}
for station in train_df['station_id'].unique():
    st[station] = pd.read_csv('../input/rrv-weather-data/1-1-16_5-31-17_Weather/'+station+'.csv', parse_dates=["calendar_date"])
    st[station]['calendar_date'] = pd.to_datetime(st[station]['calendar_date'])
    st[station]['calendar_date'] = st[station]['calendar_date'].dt.date
    st[station].rename(columns={'calendar_date':'visit_date'}, inplace=True)
    st[station]['station_id'] = station

sample_df['weather_prec'] = 0
train_df['weather_prec'] = 0
sample_df['weather_avgtemp'] = 0
train_df['weather_avgtemp'] = 0
sample_df['weather_sunlight'] = 0
train_df['weather_sunlight'] = 0
for station in train_df['station_id'].unique():
    train_df = pd.merge(train_df, st[station][['station_id','visit_date','precipitation','avg_temperature','hours_sunlight']], on=['station_id','visit_date'], how='left')
    
    train_df.loc[train_df.station_id == station, 'weather_prec'] = train_df.loc[train_df.station_id == station, 'precipitation']
    train_df.loc[train_df.station_id == station, 'weather_avgtemp'] = train_df.loc[train_df.station_id == station, 'avg_temperature']
    train_df.loc[train_df.station_id == station, 'weather_sunlight'] = train_df.loc[train_df.station_id == station, 'hours_sunlight']
    
    train_df.drop('precipitation', axis=1, inplace=True)
    train_df.drop('avg_temperature', axis=1, inplace=True)
    train_df.drop('hours_sunlight', axis=1, inplace=True)
    
    sample_df = pd.merge(sample_df, st[station][['station_id','visit_date','precipitation','avg_temperature','hours_sunlight']], on=['station_id','visit_date'], how='left')
    
    sample_df.loc[sample_df.station_id == station, 'weather_prec'] = sample_df.loc[sample_df.station_id == station, 'precipitation']
    sample_df.loc[sample_df.station_id == station, 'weather_avgtemp'] = sample_df.loc[sample_df.station_id == station, 'avg_temperature']
    sample_df.loc[sample_df.station_id == station, 'weather_sunlight'] = sample_df.loc[sample_df.station_id == station, 'hours_sunlight']
    
    sample_df.drop('precipitation', axis=1, inplace=True)
    sample_df.drop('avg_temperature', axis=1, inplace=True)
    sample_df.drop('hours_sunlight', axis=1, inplace=True)
    
#sample_df['weather_prec_prev'] = sample_df['weather_prec'].shift()
#train_df['weather_prec_prev'] = train_df['weather_prec'].shift()
#sample_df['weather_avgtemp_prev'] = sample_df['weather_avgtemp'].shift()
#train_df['weather_avgtemp_prev'] = train_df['weather_avgtemp'].shift()
#sample_df['weather_sunlight_prev'] = sample_df['weather_sunlight'].shift()
#train_df['weather_sunlight_prev'] = train_df['weather_sunlight'].shift()
    

##################################### Reservations processing

reserve_flag = 1

if reserve_flag:

    airR_df = pd.read_csv('../input/air_reserve.csv', parse_dates=["visit_datetime","reserve_datetime"])
    airR_df['visit_datetime'] = airR_df['visit_datetime'].dt.date
    airR_df['reserve_datetime'] = airR_df['reserve_datetime'].dt.date
    airR_df['reserve_date_diff'] = airR_df.apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    
    # Exclude last 5 days reservations (better than 1 week like seen in other kernel according to my tests)
    airR_df = airR_df.loc[airR_df['reserve_date_diff']>=5]
    
    airR_df1 = airR_df.groupby(['air_store_id','visit_datetime']).agg({'reserve_visitors':[np.sum,np.mean],'reserve_date_diff':[np.sum,np.mean]}).reset_index()
    airR_df1.columns = ['air_store_id', 'visit_date', 'airR_visit_sum', 'airR_visit_mean', 'airR_date_diff_sum', 'airR_date_diff_mean']
    # long-term reservations feature (useless I suppose)
    airR_df2 = airR_df.loc[airR_df['reserve_date_diff']>=40].groupby(['air_store_id','visit_datetime'])['reserve_visitors'].sum().reset_index()
    airR_df2.columns = ['air_store_id', 'visit_date', 'airR_visitors40']

    train_df = pd.merge(train_df, airR_df1, on=['air_store_id','visit_date'], how='left')
    sample_df = pd.merge(sample_df, airR_df1, on=['air_store_id','visit_date'], how='left')
    train_df = pd.merge(train_df, airR_df2, on=['air_store_id','visit_date'], how='left')
    sample_df = pd.merge(sample_df, airR_df2, on=['air_store_id','visit_date'], how='left')


    hpgR_df = pd.read_csv('../input/hpg_reserve.csv', parse_dates=["visit_datetime","reserve_datetime"])
    airhpg_df = pd.read_csv('../input/store_id_relation.csv', quotechar="'")
    hpgR_df = pd.merge(airhpg_df, hpgR_df, on='hpg_store_id', how='left')

    hpgR_df['visit_datetime'] = hpgR_df['visit_datetime'].dt.date
    hpgR_df['reserve_datetime'] = hpgR_df['reserve_datetime'].dt.date
    hpgR_df['reserve_date_diff'] = hpgR_df.apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    
    hpgR_df = hpgR_df.loc[hpgR_df['reserve_date_diff']>=5]
    
    hpgR_df1 = hpgR_df.groupby(['air_store_id','visit_datetime']).agg({'reserve_visitors':[np.sum,np.mean],'reserve_date_diff':[np.sum,np.mean]}).reset_index()
    hpgR_df1.columns = ['air_store_id', 'visit_date', 'hpgR_visit_sum', 'hpgR_visit_mean', 'hpgR_date_diff_sum', 'hpgR_date_diff_mean']
    hpgR_df2 = hpgR_df.loc[hpgR_df['reserve_date_diff']>=40].groupby(['air_store_id','visit_datetime'])['reserve_visitors'].sum().reset_index()
    hpgR_df2.columns = ['air_store_id', 'visit_date', 'hpgR_visitors40']

    train_df = pd.merge(train_df, hpgR_df1, on=['air_store_id','visit_date'], how='left')
    sample_df = pd.merge(sample_df, hpgR_df1, on=['air_store_id','visit_date'], how='left')
    train_df = pd.merge(train_df, hpgR_df2, on=['air_store_id','visit_date'], how='left')
    sample_df = pd.merge(sample_df, hpgR_df2, on=['air_store_id','visit_date'], how='left')

    # air + hpg reservations
    train_df['tot_reserve_sum'] = train_df['airR_visit_sum'] + train_df['hpgR_visit_sum']
    sample_df['tot_reserve_sum'] = sample_df['airR_visit_sum'] + sample_df['hpgR_visit_sum']
    train_df['tot_reserve_mean'] = (train_df['airR_visit_mean'] + train_df['hpgR_visit_mean']) /2
    sample_df['tot_reserve_mean'] = (sample_df['airR_visit_mean'] + sample_df['hpgR_visit_mean']) /2
    train_df['tot_date_diff_mean'] = (train_df['airR_date_diff_mean'] + train_df['hpgR_date_diff_mean']) /2
    sample_df['tot_date_diff_mean'] = (sample_df['airR_date_diff_mean'] + sample_df['hpgR_date_diff_mean']) /2
    train_df['reserve_visitors40'] = train_df['airR_visitors40'] + train_df['hpgR_visitors40']
    sample_df['reserve_visitors40'] = sample_df['airR_visitors40'] + sample_df['hpgR_visitors40']


# extra features
train_df['lon_plus_lat'] = train_df['longitude'] + train_df['latitude'] 
sample_df['lon_plus_lat'] = sample_df['longitude'] + sample_df['latitude']

lbl = LabelEncoder()
train_df['air_store_id2'] = lbl.fit_transform(train_df['air_store_id'])
sample_df['air_store_id2'] = lbl.transform(sample_df['air_store_id'])

train_df = train_df.fillna(-1)
sample_df = sample_df.fillna(-1)

####### removal of january and february 2016 from train set considering the particular wrong rolling values for these months
train_df = train_df.loc[train_df.visit_date>=datetime.datetime.strptime('2016-03-01', '%Y-%m-%d').date()]

# final train set
train_df
train_df.visitors = train_df.visitors.map(np.log1p)

x_val = train_df[train_df.visit_date >=datetime.datetime.strptime('2017-03-12', '%Y-%m-%d').date()]
x_train = train_df[train_df.visit_date < datetime.datetime.strptime('2017-03-12', '%Y-%m-%d').date()]
x_test = sample_df

params6 = {
    'learning_rate': 0.02,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}

del sample_df['station_id']
del train_df['station_id']
x_val = train_df[train_df.visit_date >=datetime.datetime.strptime('2017-03-12', '%Y-%m-%d').date()]
x_train = train_df[train_df.visit_date < datetime.datetime.strptime('2017-03-12', '%Y-%m-%d').date()]
x_test = sample_df

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
def RMSLE(y, pred):
    return mean_squared_error(y, pred)**0.5


def do_train_lgb(train_df,test_df,val, lgb_params, rounds,istc = False):
    train_df.index = range(train_df.shape[0])
    cate_vars = ['air_genre_name','holiday_flg']
    X_t = train_df.drop(['visitors','visit_date','is_outlier','air_store_id'], axis=1)
    y_t = train_df['visitors'].values
    d_train = lgb.Dataset(X_t, y_t,categorical_feature=cate_vars)
    X_v = val[X_t.columns]
    y_v = val['visitors'].values
    d_val = lgb.Dataset(X_v, y_v,categorical_feature=cate_vars)
    lgb_model = lgb.train(
        lgb_params, d_train,valid_sets=[d_train,d_val],early_stopping_rounds = 500, num_boost_round=rounds,verbose_eval = 100
    )
    val_pred = lgb_model.predict(X_v, num_iteration=lgb_model.best_iteration)
    rmsle = RMSLE(y_v, val_pred)
    print('alldf:',rmsle)
    if istc:
        test_pred = lgb_model.predict(test_df[X_t.columns], num_iteration=lgb_model.best_iteration)
        return rmsle,test_pred,val_pred,lgb_model
    X_all = pd.concat([train_df,val])
    X_t = X_all.drop(['visitors','visit_date','is_outlier','air_store_id'], axis=1)
    y_t = X_all['visitors'].values
    dtrain = lgb.Dataset(X_t, y_t,categorical_feature=cate_vars)
    lgb_model = lgb.train(
        lgb_params, dtrain,valid_sets=[dtrain], num_boost_round=lgb_model.best_iteration or rounds,verbose_eval = 100
    )
    test_pred = lgb_model.predict(test_df[X_t.columns], num_iteration=lgb_model.best_iteration)
    return rmsle,test_pred,val_pred,lgb_model

rmsle, test_pred1,val_pred1,lgb_model = do_train_lgb(x_train,x_test,x_val, params6, 300000)

x_test['pred2'] = test_pred1
testpred = x_test[['air_store_id', 'visit_date', 'pred2']]
testpred['visitors'] = testpred.pred2
testpred['id'] = testpred.air_store_id + '_' + testpred.visit_date.astype(str)
testpred['visitors'] = np.expm1(testpred.visitors)
testpred[['id','visitors']].to_csv('dndaymzy.csv',index=False)
