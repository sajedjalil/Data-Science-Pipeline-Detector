"""
Feature Engineering is heavily borrowed from Surprise Me. Thanks !
Weather Code is commented as the weather files are not part of standard dataset. With weather
information, this script gives .481 on Public LB.

"""
# DATA PREPARATION STAGE
import glob, re
import numpy as np
import pandas as pd
from sklearn import *
from datetime import datetime

###################################################################################################
#"START HKLEE FEATURE"
dfs = { re.search('/([^/\.]*)\.csv', fn).group(1):
    pd.read_csv(fn)for fn in glob.glob('../input/*.csv')}
for k, v in dfs.items(): locals()[k] = v
                     
date_info =  pd.read_csv('../input/date_info.csv')
wkend_holidays = date_info.apply(lambda x: (x.day_of_week=='Sunday' or x.day_of_week=='Saturday') and x.holiday_flg==1, axis=1)
date_info.loc[wkend_holidays, 'holiday_flg'] = 0
date_info['weight'] = ((date_info.index + 1) / len(date_info)) ** 5  

air_visit_data =  pd.read_csv('../input/air_visit_data.csv')
visit_data = air_visit_data.merge(date_info, left_on='visit_date', right_on='calendar_date', how='left')
visit_data.drop('calendar_date', axis=1, inplace=True)
visit_data['visitors'] = visit_data.visitors.map(pd.np.log1p)

visitors = visit_data.groupby(['air_store_id', 'day_of_week', 'holiday_flg']).apply(lambda x:( (x.weight * x.visitors).sum() / x.weight.sum() )).reset_index()
visitors.rename(columns={0:'visitors'}, inplace=True) 

sample_submission =  pd.read_csv('../input/sample_submission.csv')
sample_submission['air_store_id'] = sample_submission.id.map(lambda x: '_'.join(x.split('_')[:-1]))
sample_submission['calendar_date'] = sample_submission.id.map(lambda x: x.split('_')[2])
sample_submission.drop('visitors', axis=1, inplace=True)
sample_submission = sample_submission.merge(date_info, on='calendar_date', how='left')
sample_submission = sample_submission.merge(visitors, on=['air_store_id', 'day_of_week', 'holiday_flg'], how='left')

missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(visitors[visitors.holiday_flg==0], on=('air_store_id', 'day_of_week'), how='left')['visitors_y'].values

missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(visitors[['air_store_id', 'visitors']].groupby('air_store_id').mean().reset_index(), on='air_store_id', how='left')['visitors_y'].values

test_visit_var = sample_submission.visitors.map(pd.np.expm1)

data = {
    'tra': pd.read_csv('../input/air_visit_data.csv')
    }

data['tra'] = data['tra'].merge(date_info, left_on='visit_date', right_on='calendar_date', how='left')
data['tra'] = data['tra'].merge(visitors, on=['air_store_id', 'day_of_week', 'holiday_flg'], how='left')

missings = data['tra'].visitors_y.isnull()
data['tra'].loc[missings, 'visitors_y'] = data['tra'][missings].merge(visitors[visitors.holiday_flg==0], on=('air_store_id', 'day_of_week'), how='left')['visitors_y'].values

missings = data['tra'].visitors_y.isnull()
data['tra'].loc[missings, 'visitors_y'] = data['tra'][missings].merge(visitors[['air_store_id', 'visitors']].groupby('air_store_id').mean().reset_index(), on='air_store_id', how='left')['visitors_y'].values

train_visit_var = data['tra'].visitors_y.map(pd.np.expm1)

###################################################################################################

data = {
    'tra': pd.read_csv('../input/air_visit_data.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/air_reserve.csv'),
    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    'id': pd.read_csv('../input/store_id_relation.csv'),
    'tes': pd.read_csv('../input/sample_submission.csv'),
    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

for df in ['ar','hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors':'rv1'})
    tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors':'rv2'})
    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date
    
# Adding golden week variable here For training
data['tra']['DT'] = pd.to_datetime(data['tra']['visit_date'])
start_date = pd.to_datetime('2016-04-28')
end_date = pd.to_datetime('2016-05-06')
mask1 = (data['tra']['DT'] > start_date) & (data['tra']['DT'] < end_date)
start_date2 = pd.to_datetime('2017-04-28')
end_date2 = pd.to_datetime('2017-05-06')
mask2 = (data['tra']['DT'] > start_date2) & (data['tra']['DT'] < end_date2)
mask = mask1 | mask2
maskint = mask.astype(int)
data['tra']['golden_week'] = maskint
data['tra']['golden_week']  = data['tra']['golden_week'].astype(int)
del(mask, mask1,mask2, maskint)
data['tra'].drop('DT', axis = 1, inplace = True)

data['tes']['DT'] = pd.to_datetime(data['tes']['visit_date'])
start_date = pd.to_datetime('2016-04-28')
end_date = pd.to_datetime('2016-05-06')
mask1 = (data['tes']['DT'] > start_date) & (data['tes']['DT'] < end_date)
start_date2 = pd.to_datetime('2017-04-28')
end_date2 = pd.to_datetime('2017-05-06')
mask2 = (data['tes']['DT'] > start_date2) & (data['tes']['DT'] < end_date2)
mask = mask1 | mask2
maskint = mask.astype(int)
data['tes']['golden_week'] = maskint
data['tes']['golden_week']  = data['tes']['golden_week'].astype(int)
del(mask, mask1,mask2, maskint)
data['tes'].drop('DT', axis = 1, inplace = True)
# End adding golden week variable.

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)
stores
#OPTIMIZED BY JEROME VALLET
tmp = data['tra'].groupby(['air_store_id','dow']).agg({'visitors' : [np.min,np.mean,np.median,np.max,np.size]}).reset_index()
tmp.columns = ['air_store_id', 'dow', 'min_visitors', 'mean_visitors', 'median_visitors','max_visitors','count_observations']
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 

stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 
stores.dtypes
# NEW FEATURES FROM Georgii Vyshnia
stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))
stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))

# Following lines of code restrict the air_genre_name and air_area_name so using ohe here and commenting the following lines.

#lbl = preprocessing.LabelEncoder()
#
#for i in range(10):
#    stores['air_genre_name'+str(i)] = lbl.fit_transform(stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
#    stores['air_area_name'+str(i)] = lbl.fit_transform(stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
#stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
#stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

## *************************** AIR GENRE NAME OHE #######################################
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
enc=OneHotEncoder(sparse=False)
labelencoder_X_air_genre_name = LabelEncoder()
stores.air_genre_name = labelencoder_X_air_genre_name.fit_transform(stores.air_genre_name)
stores
tmp_data= stores['air_genre_name']
tmp_data = tmp_data.reshape(-1, 1)
var_ohe = pd.DataFrame(enc.fit_transform(tmp_data))
var_ohe = var_ohe.add_prefix("air_genre_name_")
var_ohe = var_ohe.astype(int)
stores = stores.join(var_ohe)
stores.drop('air_genre_name', axis=1, inplace=True)
del(tmp_data,var_ohe,enc)
# *************************** END AIR GENRE NAME OHE ###################################

## *************************** AIR AREA NAME OHE #######################################
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
enc=OneHotEncoder(sparse=False)
labelencoder_X_air_genre_name = LabelEncoder()
stores.air_area_name = labelencoder_X_air_genre_name.fit_transform(stores.air_area_name)
tmp_data= stores['air_area_name']
tmp_data = tmp_data.reshape(-1, 1)
var_ohe = pd.DataFrame(enc.fit_transform(tmp_data))
var_ohe = var_ohe.add_prefix("air_area_name_")
var_ohe = var_ohe.astype(int)
stores = stores.join(var_ohe)
stores.drop('air_area_name', axis=1, inplace=True)
del(tmp_data,var_ohe,enc)
# *************************** END AIR AREA NAME OHE ###################################

lbl = preprocessing.LabelEncoder()
data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 

train = pd.merge(train, stores, how='left', on=['air_store_id','dow']) 
test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])

for df in ['ar','hr']:
    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 
    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])

train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)

train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2

test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2

# NEW FEATURES FROM JMBULL
train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

# splitting date_int to three different variables day month and year.
train['pyear'], train['pmonth'], train['pday'] = train['date_int'].astype(str).str[:4].astype(int), train['date_int'].astype(str).str[4:6].astype(int), np.log1p(train['date_int'].astype(str).str[6:8].astype(int))
test['pyear'], test['pmonth'], test['pday'] = test['date_int'].astype(str).str[:4].astype(int), test['date_int'].astype(str).str[4:6].astype(int), np.log1p(test['date_int'].astype(str).str[6:8].astype(int))

train.drop('date_int', axis=1, inplace=True)
test.drop('date_int', axis=1, inplace=True)


train['var_max_lat'] = train['latitude'].max() - train['latitude']
train['var_max_long'] = train['longitude'].max() - train['longitude']
test['var_max_lat'] = test['latitude'].max() - test['latitude']
test['var_max_long'] = test['longitude'].max() - test['longitude']

# NEW FEATURES FROM Georgii Vyshnia
train['lon_plus_lat'] = train['longitude'] + train['latitude'] 
test['lon_plus_lat'] = test['longitude'] + test['latitude']

train.visit_date = pd.to_datetime(train.visit_date)
test.visit_date = pd.to_datetime(test.visit_date)
# **********************************  WEATHER DATA ************************************#
def add_weather(dataset):                                                                                                                     
    print('Adding weather...')
                                                                                                     
    air_nearest = pd.read_csv( './input/weather/air_store_info_with_nearest_active_station.csv')                                                              
    unique_air_store_ids = list(dataset.air_store_id.unique())                                                                                
                                                                                                                                              
    weather_dir = './input/weather/1-1-16_5-31-17_Weather/'                                                                            
    weather_keep_columns = ['precipitation', 'avg_temperature','high_temperature','low_temperature','hours_sunlight','avg_wind_speed']                                                                                                                         
                                                                                                                                              
    dataset_with_weather = dataset.copy()                                                                                                     
    for column in weather_keep_columns:                                                                                                       
        dataset_with_weather[column] = np.nan                                                                                                 
                                                                                                                                              
    for air_id in unique_air_store_ids:                                                                                                       
        station = air_nearest[air_nearest.air_store_id == air_id].station_id.iloc[0]                                                          
        weather_data = pd.read_csv(weather_dir + station + '.csv', parse_dates=['calendar_date']).rename(columns={'calendar_date': 'visit_date'})                                                                                                                                           
                                                                                                                                              
        this_store = dataset.air_store_id == air_id                                                                                           
        merged = dataset[this_store].merge(weather_data, on='visit_date', how='left')                                                         
                                                                                                                                              
        for column in weather_keep_columns:                                                                                                   
            dataset_with_weather.loc[this_store, column] = merged[column]                                                         
    return dataset_with_weather                                                                                                               
                                                                                                                                              
#train = add_weather(train)   # Commented                                                                                                                 
#test = add_weather(test)   # Commented
# *****************************     End weather data *****************************#

# **************************Adding day month and year*****************************#
train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
train['pyear'], train['pmonth'], train['pday'] = train['date_int'].astype(str).str[:4].astype(int), train['date_int'].astype(str).str[4:6].astype(int), np.log1p(train['date_int'].astype(str).str[6:8].astype(int))
test['pyear'], test['pmonth'], test['pday'] = test['date_int'].astype(str).str[:4].astype(int), test['date_int'].astype(str).str[4:6].astype(int), np.log1p(test['date_int'].astype(str).str[6:8].astype(int))
train.drop('date_int', axis=1, inplace=True)
test.drop('date_int', axis=1, inplace=True)

# End day month and year

lbl = preprocessing.LabelEncoder()
train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
test['air_store_id2'] = lbl.transform(test['air_store_id'])

train['hklee_feature'] = train_visit_var 
test['hklee_feature'] = test_visit_var
    
# OHE FOR DOW #######

# END OHE FOR DOW ###    
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lbldow = preprocessing.LabelEncoder()
train['dow2'] = lbldow.fit_transform(train['dow'])
test['dow2'] = lbldow.transform(test['dow'])
enc=OneHotEncoder(sparse=False)
tmp_data= train['dow2']
tmp_data = tmp_data.reshape(-1, 1)
var_ohe = pd.DataFrame(enc.fit_transform(tmp_data))
var_ohe = var_ohe.add_prefix("dow2_")
train = train.join(var_ohe)
train.drop('dow2', axis=1, inplace=True)
train.drop('dow', axis=1, inplace=True)
tmp_data= test['dow2']
tmp_data = tmp_data.reshape(-1, 1)
var_ohe = pd.DataFrame(enc.transform(tmp_data))
var_ohe = var_ohe.add_prefix("dow2_")
test = test.join(var_ohe)
test.drop('dow2', axis=1, inplace=True)
test.drop('dow', axis=1, inplace=True)
train['RITESH'] = 'train'
test['RITESH'] = 'submission'
df_dataset_prep = train.append(test)
df_dataset = df_dataset_prep.copy()
del(df_dataset_prep)
#df_dataset_prep.to_pickle('df_dataset_prep.pickle')
# END DATASET CREATION STAGE

# XGBOOST MODELLING 

#df_dataset_prep = pd.read_pickle('df_submission.pickle')
df_dataset['month'] = np.sqrt(df_dataset.month)
df_dataset['max_visitors'] = np.sqrt(np.abs(df_dataset.max_visitors))
df_dataset['mean_visitors'] = np.sqrt(np.abs(df_dataset.mean_visitors))
df_dataset['median_visitors'] = np.sqrt(np.abs(df_dataset.median_visitors))
df_dataset['hklee_feature'] = np.sqrt(np.abs(df_dataset.hklee_feature))
#df_dataset['avg_temperature'] = np.log1p(df_dataset.avg_temperature) # 
#df_dataset['tempdiff'] = np.sqrt(np.abs(df_dataset.high_temperature - df_dataset.low_temperature))  #
#df_dataset['ratio_tempdiff_avg_temperature'] = df_dataset.tempdiff/df_dataset.avg_temperature  # removed in 23
df_dataset['rv1_x'] = np.sqrt(np.abs(df_dataset.rv1_x))
df_dataset['latitude'] = np.sqrt(np.abs(df_dataset.latitude))
df_dataset['longitude'] = np.sqrt(np.abs(df_dataset.longitude))
df_dataset['pyear'] = np.sqrt(np.abs(df_dataset.pyear))
df_dataset['count_observations'] = np.sqrt(np.abs(df_dataset.count_observations))
df_dataset.drop('pmonth', axis=1, inplace=True)
df_dataset.dtypes.to_csv("dtypes-temp.csv")
df_dataset['cmb_latlong'] = df_dataset['latitude'].astype(str) + df_dataset['longitude'].astype(str)
df_dataset['cmb_latlong'] = df_dataset.cmb_latlong.astype(str)
# adding OHE for LATLONG #############################################
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
enc=OneHotEncoder(sparse=False)
labelencoder_X = LabelEncoder()
df_dataset.cmb_latlong = labelencoder_X.fit_transform(df_dataset.cmb_latlong)
tmp_data= df_dataset['cmb_latlong']
tmp_data = tmp_data.reshape(-1, 1)
var_ohe = pd.DataFrame(enc.fit_transform(tmp_data))
var_ohe = var_ohe.add_prefix("cmb_latlong_")
var_ohe = var_ohe.astype(int)
df_dataset = df_dataset.join(var_ohe)
df_dataset.drop('cmb_latlong', axis=1, inplace=True)
del(tmp_data,var_ohe,enc)
# End OHE for lATLONG ################################################
## START df_dataset feature engineering do here        **********************************************#
## END df_dataset feature engineering do here    ***************************************************#
train_items = df_dataset.loc[df_dataset['RITESH'] == 'train']
train_items.drop('RITESH', axis=1, inplace=True)
test = df_dataset.loc[df_dataset['RITESH'] == 'submission']
test.drop('RITESH', axis=1, inplace=True)


def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

from sklearn.model_selection import train_test_split
train_items.drop('air_store_id' , axis=1, inplace=True)
train_items['visit_date'] = pd.to_datetime(train_items.visit_date)
y = np.log1p(train_items.visitors)
train_items.drop('visitors', axis=1, inplace=True)
train_items.drop('visit_date' , axis=1, inplace=True)
train_items.drop('id' , axis=1, inplace=True)
train_items.dtypes.to_csv("modelling-6-data.csv")
import xgboost as xgb
X_dtrain, X_deval, y_dtrain, y_deval =    train_test_split(train_items, y, random_state=1026, test_size=0.0001)
dtrain = xgb.DMatrix(X_dtrain, y_dtrain)
dvalid = xgb.DMatrix(X_deval, y_deval)
target = 'visitors'
features = [x for x in (train_items.columns.values) if x not in [target]]
best_parameters = {'colsample_bytree': 0.65, 'gamma': 1, 'learning_rate': 0.03, 'max_depth': 11, 'min_child_weight': 7, 'n_estimators': 550, 'nthread': 4, 'objective': 'reg:linear', 'reg_alpha': 0, 'silent': 0, 'subsample': 0.9}
num_boost_round = 5000
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
print("XGBoost Modelling started.")
gbm = xgb.train(best_parameters, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=200, verbose_eval=True) # 
print("Validating")
yhat = gbm.predict(xgb.DMatrix(X_deval))
error = rmspe(np.expm1(y_deval), np.expm1(yhat))
print('RMSPE: {:.6f}'.format(error)) 

test_predictions = np.expm1(gbm.predict(xgb.DMatrix(test[features])))
#feat_imp = pd.Series(gbm.get_fscore()).sort_values(ascending=False)
#feat_imp.plot(kind='bar', title='Feature Importance')                   
test['visitors'] = test_predictions
sub1 = test[['id','visitors']].copy()
sub1.to_csv("submissionxyz.csv", index=False) # 




