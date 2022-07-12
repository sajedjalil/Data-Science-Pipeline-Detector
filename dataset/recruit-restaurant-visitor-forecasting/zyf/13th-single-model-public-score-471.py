import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime
import lightgbm as lgb
import xgboost as xgb
from datetime import date,timedelta
from sklearn import preprocessing, metrics
import  sklearn.model_selection as model_selection
from sklearn.cross_validation import KFold
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import warnings

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# os.chdir("/home/viczyf/kaggle/recruit")
os.chdir("F:/pythonfiles/kaggle/recruit")

df_air_visit_data = pd.read_csv('data/air_visit_data.csv')
df_air_visit_data['visitors']=df_air_visit_data['visitors'].apply(np.log1p)
df_air_store_info = pd.read_csv('data/air_store_info.csv')
# df_hpg_store_info = pd.read_csv('data/hpg_store_info.csv')
df_store_id_relation = pd.read_csv('data/store_id_relation.csv')
df_sample_submission = pd.read_csv('data/sample_submission.csv')
df_date_info = pd.read_csv('data/date_info.csv').rename(columns={'calendar_date':'visit_date'})
df_date_info.loc[(df_date_info['day_of_week']=='Saturday')|(df_date_info['day_of_week']=='Sunday'),'holiday_flg']=0
df_air_reserve = pd.read_csv('data/air_reserve.csv')
df_hpg_reserve = pd.read_csv('data/hpg_reserve.csv')
df_hpg_reserve = pd.merge(df_hpg_reserve,df_store_id_relation,how='inner',on=['hpg_store_id'])

start_date=df_air_visit_data.groupby(['air_store_id'],as_index=False)['visit_date'].first().rename(columns={'visit_date':"start_date"})
start_date['start_date'] = pd.to_datetime(start_date['start_date'])

df_air_visit_data['visit_date'] = pd.to_datetime(df_air_visit_data['visit_date'])
df_air_visit_data['dow'] = df_air_visit_data['visit_date'].dt.dayofweek
df_air_visit_data['year'] = df_air_visit_data['visit_date'].dt.year
df_air_visit_data['month'] = df_air_visit_data['visit_date'].dt.month
df_air_visit_data['day'] = df_air_visit_data['visit_date'].dt.day
df_air_visit_data['visit_date'] = df_air_visit_data['visit_date'].dt.date
df_air_visit_data['month_day']=df_air_visit_data['month']*100+df_air_visit_data['day']

df_sample_submission['visit_date'] = df_sample_submission['id'].map(lambda x: str(x).split('_')[2])
df_sample_submission['air_store_id'] = df_sample_submission['id'].map(lambda x: '_'.join(x.split('_')[:2]))
df_sample_submission['visit_date'] = pd.to_datetime(df_sample_submission['visit_date'])
df_sample_submission['dow'] = df_sample_submission['visit_date'].dt.dayofweek
df_sample_submission['year'] = df_sample_submission['visit_date'].dt.year
df_sample_submission['month'] = df_sample_submission['visit_date'].dt.month
df_sample_submission['day'] = df_sample_submission['visit_date'].dt.day
df_sample_submission['visit_date'] = df_sample_submission['visit_date'].dt.date
df_sample_submission['month_day']=df_sample_submission['month']*100+df_sample_submission['day']
# 做一个39天的标记
df_sample_submission['mark_day']=(df_sample_submission['month']-4)*7+df_sample_submission['day']
df_sample_submission.loc[df_sample_submission['month']==4,'mark_day']=df_sample_submission.loc[df_sample_submission['month']==4,'day']-23

# 计算训练集中 测试集中出现过的店铺 每个曜日的顾客数量的统计值
unique_stores = df_sample_submission['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i] * len(unique_stores)}) for i in range(7)],
                   axis=0, ignore_index=True).reset_index(drop=True)

# sure it can be compressed...
tmp = df_air_visit_data.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].min().rename(columns={'visitors': 'dow_min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = df_air_visit_data.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].mean().rename(columns={'visitors': 'dow_mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = df_air_visit_data.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].median().rename(columns={'visitors': 'dow_median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = df_air_visit_data.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].max().rename(columns={'visitors': 'dow_max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = df_air_visit_data.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].count().rename(columns={'visitors': 'dow_count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = df_air_visit_data.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].var().rename(columns={'visitors': 'dow_variation_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = df_air_visit_data.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].skew().reset_index().rename(columns={0: 'dow_skew_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = df_air_visit_data.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].apply(kurtosis).reset_index().rename(columns={0: 'dow_kurto_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = df_air_visit_data.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].quantile(0.1).reset_index().rename(columns={0: 'dow_quantile1_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = df_air_visit_data.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].quantile(0.9).reset_index().rename(columns={0: 'dow_quantile9_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = df_air_visit_data.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].sum().rename(columns={'visitors': 'dow_sum_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])

tmp = df_air_visit_data.groupby(['air_store_id'], as_index=False)['visitors'].min().rename(columns={'visitors': 'store_min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id'])
tmp = df_air_visit_data.groupby(['air_store_id'], as_index=False)['visitors'].mean().rename(columns={'visitors': 'store_mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id'])
tmp = df_air_visit_data.groupby(['air_store_id'], as_index=False)['visitors'].median().rename(columns={'visitors': 'store_median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id'])
tmp = df_air_visit_data.groupby(['air_store_id'], as_index=False)['visitors'].max().rename(columns={'visitors': 'store_max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id'])
tmp = df_air_visit_data.groupby(['air_store_id'], as_index=False)['visitors'].count().rename(columns={'visitors': 'store_count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id'])
tmp = df_air_visit_data.groupby(['air_store_id'], as_index=False)['visitors'].var().rename(columns={'visitors': 'store_variation_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id'])
tmp = df_air_visit_data.groupby(['air_store_id'])['visitors'].quantile(0.1).reset_index().rename(columns={'visitors' : 'store_quantile1_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id'])
tmp = df_air_visit_data.groupby(['air_store_id'])['visitors'].quantile(0.9).reset_index().rename(columns={'visitors' : 'store_quantile9_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id'])

tmp = df_air_visit_data.groupby(['air_store_id'], as_index=False)['visitors'].sum().rename(columns={'visitors': 'sum_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id'])
stores['dow_visitors_ratio'] = stores['dow_sum_visitors'] / stores['sum_visitors']

#  merge上air_store_info的信息
stores = pd.merge(stores, df_air_store_info, how='left', on=['air_store_id'])

# NEW FEATURES FROM Georgii Vyshnia
# 但是我感觉我这样的处理方式 还不如直接one-hot或者编码
# 搞错了，歧视这里是 类似 市，县，乡 这样的划分
stores['area_len']=stores['air_area_name'].apply(lambda x:len(x.split(" ")))
stores.loc[stores['area_len']>3,'air_area_name'].unique()
map_area={
       'Hokkaidō Sapporo-shi Kotoni 2 Jō' : 'Hokkaidō Sapporo-shi KotoniJō',
       'Tōkyō-to Musashino-shi Kichijōji Honchō' : 'Tōkyō-to Musashino-shi KichijōjiHonchō',
       'Fukuoka-ken Fukuoka-shi Hakata Ekimae' : 'Fukuoka-ken Fukuoka-shi HakataEkimae',
       'Hyōgo-ken Kakogawa-shi Kakogawachō Kitazaike' : 'Hyōgo-ken Kakogawa-shi KakogawachōKitazaike',
       'Hokkaidō Sapporo-shi Minami 3 Jōnishi' : 'Hokkaidō Sapporo-shi MinamiJōnishi',
       'Niigata-ken Niigata-shi Gakkōchōdōri 1 Banchō' : 'Niigata-ken Niigata-shi GakkōchōdōriBanchō',
       'Tōkyō-to Chiyoda-ku Kanda Jinbōchō' : 'Tōkyō-to Chiyoda-ku KandaJinbōchō',
       'Hokkaidō Asahikawa-shi 6 Jōdōri' : 'Hokkaidō Asahikawa-shi Jōdōri',
       'Hokkaidō Abashiri-shi Minami 6 Jōhigashi' : 'Hokkaidō Abashiri-shi MinamiJōhigashi',
       'Hyōgo-ken Kōbe-shi Sumiyoshi Higashimachi' : 'Hyōgo-ken Kōbe-shi SumiyoshiHigashimachi',
       'Hokkaidō Sapporo-shi Kita 24 Jōnishi' : 'Hokkaidō Sapporo-shi Kita24Jōnishi',
       'Hokkaidō Sapporo-shi Atsubetsuchūō 1 Jō' : 'Hokkaidō Sapporo-shi AtsubetsuchūōJō'
}
stores.loc[stores['area_len']>3,'air_area_name']=stores.loc[stores['area_len']>3,'air_area_name'].map(map_area)
del stores['area_len']
lbl = preprocessing.LabelEncoder()
stores=pd.get_dummies(stores,columns=['air_genre_name'])
for i in range(3):
    stores['air_area_name' + str(i)] = lbl.fit_transform(stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' ')) > i else ''))
stores['air_area_name'] = stores['air_area_name0']*10000+stores['air_area_name1']*100+stores['air_area_name2']

def rmse(y_true,y_pred):
    return np.sqrt(metrics.mean_squared_error(y_true,y_pred))
def rmse_cv(model, X_train, y):
    # RMSE with Cross Validation
    rmse= np.sqrt(-model_selection.cross_val_score(model, X_train, y,scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
def get_reserve_timeandvisitors(df,day):
    df['visit_datetime'] = pd.to_datetime(df['visit_datetime'])
    df['reserve_datetime'] = pd.to_datetime(df['reserve_datetime'])
    # 选择预订时间在访问时间n天之前的
    df['visit_datetime_date']=df['visit_datetime'].dt.date
    df['reserve_datetime_date']=df['reserve_datetime'].dt.date
    df['reserve_beforevisit_days']=df.apply(lambda r: (r['visit_datetime_date'] - r['reserve_datetime_date']).days,axis=1)
    df=df.loc[df['reserve_beforevisit_days']>=day+1]
    df['reserve_datetime_diff'] = (df['visit_datetime'].dt.hour-df['reserve_datetime'].dt.hour)+\
                                  df.apply(lambda r: (r['visit_datetime_date'] - r['reserve_datetime_date']).days,axis=1)*24
    # df['reserve_datetime_diff'] =df.apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).hour,axis=1)
    # df['reserve_datetime_diff'] = df['visit_datetime'].dt.hour - df['reserve_datetime'].dt.hour
    df['visit_datetime'] = df['visit_datetime'].dt.date
    df['reserve_datetime'] = df['reserve_datetime'].dt.date
    tmp1 = df.groupby(['air_store_id', 'visit_datetime'], as_index=False)[
        ['reserve_datetime_diff', 'reserve_visitors']].sum().rename(
        columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors': 'rv1'})
    tmp2 = df.groupby(['air_store_id', 'visit_datetime'], as_index=False)[
        ['reserve_datetime_diff', 'reserve_visitors']].mean().rename(
        columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors': 'rv2'})
    df = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id', 'visit_date'])
    return df
def add_weather(dataset):
    # https: // www.kaggle.com / supermdguy / using - the - weather - data / notebook
    # print('Adding weather...')
    air_nearest = pd.read_csv('data/weather/air_store_info_with_nearest_active_station.csv')
    unique_air_store_ids = list(dataset.air_store_id.unique())
    weather_dir = 'data/weather/1-1-16_5-31-17_Weather/'
    weather_keep_columns = ['precipitation', 'avg_temperature' , 'avg_wind_speed' ,'avg_humidity']
    dataset_with_weather = dataset.copy()
    for column in weather_keep_columns:
        dataset_with_weather[column] = np.nan

    for air_id in unique_air_store_ids:
        # air_id='air_24e8414b9b07decb'
        station = air_nearest[air_nearest.air_store_id == air_id].station_id.iloc[0]
        # print(station)
        weather_data = pd.read_csv(weather_dir + station + '.csv', parse_dates=['calendar_date']).rename(
            columns={'calendar_date': 'visit_date'})
        weather_data['visit_date']=weather_data['visit_date'].dt.date
        this_store = dataset.air_store_id == air_id
        merged = dataset[this_store].merge(weather_data, on='visit_date', how='left')
        for column in weather_keep_columns:
            dataset_with_weather.loc[this_store, column] = merged[column].values
    return dataset_with_weather

# day=2
cv_score=[]
for day in range(39):
# for day in [2,10,28]:
# for day in [2,8,13,18,24,29,35]:
    air_reserve=get_reserve_timeandvisitors(df_air_reserve,day)
    hpg_reserve=get_reserve_timeandvisitors(df_hpg_reserve,day)
    air_visit_data=df_air_visit_data.copy()
    sample_submission=df_sample_submission.copy()
    date_info=df_date_info.copy()
    # 前28天的记录
    air_visit_data['base_date']=pd.to_datetime(air_visit_data['visit_date'])
    sample_submission['base_date']=pd.to_datetime(sample_submission['visit_date'])

    for i in range(1,29):
        tmp = air_visit_data.copy()
        tmp['visit_date'] = pd.to_datetime(tmp['visit_date'])
        tmp['base_date'] = tmp['visit_date'] + timedelta(day+i)
        tmp = tmp.rename(columns={"visitors": "before_{}days_visitors".format(i)})
        # tmp['visit_date'] = tmp['visit_date'].dt.date
        air_visit_data = pd.merge(air_visit_data, tmp[['air_store_id', 'base_date', "before_{}days_visitors".format(i)]], how='left',on=['air_store_id', 'base_date'])
        sample_submission = pd.merge(sample_submission, tmp[['air_store_id', 'base_date', "before_{}days_visitors".format(i)]],how='left',on=['air_store_id', 'base_date'])

    for i in [7,14,21,28]:
        # print(i)
        col_list=[]
        for x in range(1,i+1):
            col_list.append( "before_{}days_visitors".format(x))
        # air_visit_data['sumnull_{}_days'.format(i)] = air_visit_data[col_list].isnull().sum(axis=1).values
        air_visit_data['sum_{}_days'.format(i)]=air_visit_data[col_list].sum(axis=1).values
        air_visit_data['mean_{}_days'.format(i)]=air_visit_data[col_list].mean(axis=1).values
        air_visit_data['decaymean_{}_days'.format(i)]=(air_visit_data[col_list]*np.power(0.9,np.arange(i))).sum(axis=1)
        air_visit_data['median_{}_days'.format(i)] = air_visit_data[col_list].median(axis=1).values
        air_visit_data['min_{}_days'.format(i)]=air_visit_data[col_list].min(axis=1).values
        air_visit_data['max_{}_days'.format(i)]=air_visit_data[col_list].max(axis=1).values
        # air_visit_data['std_{}_days'.format(i)]=air_visit_data[col_list].std(axis=1).values

        # sample_submission['sumnull_{}_days'.format(i)] = sample_submission[col_list].isnull().sum(axis=1).values
        sample_submission['sum_{}_days'.format(i)]=sample_submission[col_list].sum(axis=1).values
        sample_submission['mean_{}_days'.format(i)]=sample_submission[col_list].mean(axis=1).values
        sample_submission['decaymean_{}_days'.format(i)]=(sample_submission[col_list]*np.power(0.9,np.arange(i))).sum(axis=1)
        sample_submission['median_{}_days'.format(i)]=sample_submission[col_list].median(axis=1).values
        sample_submission['min_{}_days'.format(i)]=sample_submission[col_list].min(axis=1).values
        sample_submission['max_{}_days'.format(i)]=sample_submission[col_list].max(axis=1).values
        # sample_submission['std_{}_days'.format(i)]=sample_submission[col_list].std(axis=1).values

    # 曜日特征呢个
    weekday_mark = day%7
    list_4weekday=["before_{}days_visitors".format(7-weekday_mark),"before_{}days_visitors".format(14-weekday_mark),"before_{}days_visitors".format(21-weekday_mark),"before_{}days_visitors".format(28-weekday_mark)]
    air_visit_data['sum4_dow_visitors'] = air_visit_data[list_4weekday].sum(axis=1)
    air_visit_data['mean4_dow_visitors'] = air_visit_data[list_4weekday].mean(axis=1)
    air_visit_data['median4_dow_visitors'] = air_visit_data[list_4weekday].median(axis=1)
    air_visit_data['min4_dow_visitors'] = air_visit_data[list_4weekday].min(axis=1)
    air_visit_data['max4_dow_visitors'] = air_visit_data[list_4weekday].max(axis=1)
    sample_submission['sum4_dow_visitors'] = sample_submission[list_4weekday].sum(axis=1)
    sample_submission['mean4_dow_visitors'] = sample_submission[list_4weekday].mean(axis=1)
    sample_submission['median4_dow_visitors'] = sample_submission[list_4weekday].median(axis=1)
    sample_submission['min4_dow_visitors'] = sample_submission[list_4weekday].min(axis=1)
    sample_submission['max4_dow_visitors'] = sample_submission[list_4weekday].max(axis=1)

    air_visit_data['dow_4week_ratio']=air_visit_data['sum4_dow_visitors']/air_visit_data['sum_28_days']
    sample_submission['dow_4week_ratio']=sample_submission['sum4_dow_visitors']/sample_submission['sum_28_days']
    date_info['visit_date'] = pd.to_datetime(date_info['visit_date'])
    # date_info['day_of_week'] = lbl.fit_transform(date_info['day_of_week'])
    # map_dow={'Monday':1,'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7 }
    # date_info['day_of_week'] = date_info['day_of_week'].map(map_dow)

    date_info['visit_date'] = date_info['visit_date'].dt.date
    date_info['lastday_isholiday_flg']=date_info['holiday_flg'].shift(periods=1).values
    date_info['nextday_isholiday_flg']=date_info['holiday_flg'].shift(periods=-1).values
    # 考虑加入 距离上一次和下一次放假的时间
    day_from_lastholiday=[]
    lastholiday_mark=0
    for index,row in date_info.iterrows():
        if index==0:
            day_from_lastholiday.append(0)
            if row.holiday_flg==1:
                lastholiday_mark=index
        else:
            day_from_lastholiday.append(index-lastholiday_mark)
            if row.holiday_flg == 1:
                lastholiday_mark = index
    date_info['day_from_lastholiday']=day_from_lastholiday
    date_info_copy=date_info.sort_values(by=['visit_date'],ascending=False).reset_index(drop=True)
    day_before_nextholiday=[]
    nextholiday_mark=-31
    for index,row in date_info_copy.iterrows():
        if index==0:
            day_before_nextholiday.append(31)
            if row.holiday_flg==1:
                nextholiday_mark=index
        else:
            day_before_nextholiday.append(index-nextholiday_mark)
            if row.holiday_flg == 1:
                nextholiday_mark = index
    day_before_nextholiday.reverse()
    date_info['day_before_nextholiday']=day_before_nextholiday
    date_info['day_dis_holiday']=date_info[['day_from_lastholiday','day_before_nextholiday']].min(axis=1)
    date_info=date_info.drop(['day_of_week'],axis=1)

    train = pd.merge(air_visit_data, date_info, how='left', on=['visit_date'])
    test = pd.merge(sample_submission, date_info, how='left', on=['visit_date'])

    train = pd.merge(train, stores, how='left', on=['air_store_id', 'dow'])
    test = pd.merge(test, stores, how='left', on=['air_store_id', 'dow'])

    for df in [air_reserve , hpg_reserve]:
        train = pd.merge(train, df, how='left', on=['air_store_id', 'visit_date'])
        test = pd.merge(test, df, how='left', on=['air_store_id', 'visit_date'])

    # train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)

    # train['total_reserv_sum'] = train[['rv1_x','rv1_y']].sum(axis=1)
    train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
    # train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
    # train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2

    # test['total_reserv_sum'] = test[['rv1_x','rv1_y']].sum(axis=1)
    test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
    # test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
    # test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2

    # NEW FEATURES FROM JMBULL
    train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    # train['var_max_lat'] = train['latitude'].max() - train['latitude']
    # train['var_max_long'] = train['longitude'].max() - train['longitude']
    # test['var_max_lat'] = test['latitude'].max() - test['latitude']
    # test['var_max_long'] = test['longitude'].max() - test['longitude']

    # NEW FEATURES FROM Georgii Vyshnia
    train['lon_plus_lat'] = train['longitude'] + train['latitude']
    test['lon_plus_lat'] = test['longitude'] + test['latitude']

    lbl = preprocessing.LabelEncoder()
    train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
    test['air_store_id2'] = lbl.transform(test['air_store_id'])

    # 试图去掉一个店铺刚开的时候 部分数据
    train=pd.merge(train,start_date,how='left',on=['air_store_id'])
    train['visit_date']=pd.to_datetime(train['visit_date'])
    train['start_date']=pd.to_datetime(train['start_date'])+timedelta(42)
    train=train.loc[train['visit_date']>=train['start_date']]
    train['visit_date']=train['visit_date'].dt.date

    train = add_weather(train)
    test = add_weather(test)

    col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date', 'visitors','week' ,'start_date', 'base_date'
                                         ,'rs1_x','rs1_y','rs2_x','rs2_y','rv2_x','rv2_y']]

    kf = KFold(train.shape[0], n_folds=5, random_state=2017)
    print("预测第{}天的数据".format(day))
    print("训练开始：", datetime.now())
    model=XGBRegressor(learning_rate=0.01, seed=3, n_estimators=1500, subsample=0.8,
                       colsample_bytree=0.8, max_depth=10)

    model.fit(train[col],train['visitors'])
    y_pred=model.predict(test.loc[test['mark_day']==day,col])
    df_sample_submission.loc[df_sample_submission['mark_day']==day,'visitors'] = np.expm1(y_pred)

df_sample_submission[['id', 'visitors']].to_csv('results/submission_model_splitday_win10_xgb.csv', index=False)
