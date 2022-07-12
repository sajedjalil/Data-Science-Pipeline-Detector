"""
A first approach on the problem

__author__ : Konstantina

"""

import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn import decomposition, pipeline, metrics, grid_search
from sklearn.svm import SVR
import sys

#read data
users = pd.read_csv('../input/user_list.csv',encoding = 'utf-8', parse_dates=['REG_DATE','WITHDRAW_DATE'])
users['PREF_NAME'] = users['PREF_NAME'].apply(lambda s: ('%s' % s).encode("utf-8"))

print("Users df")
print(users.columns.to_series().groupby(users.dtypes).groups)
print(users.shape)

coupons_train = pd.read_csv('../input/coupon_list_train.csv', parse_dates=['DISPFROM','DISPEND','VALIDFROM','VALIDEND'])
coupons_train['CAPSULE_TEXT'] = coupons_train['CAPSULE_TEXT'].apply(lambda s: ('%s' % s).encode("utf-8"))
coupons_train['GENRE_NAME'] = coupons_train['GENRE_NAME'].apply(lambda s: ('%s' % s).encode("utf-8"))
coupons_train['large_area_name'] = coupons_train['large_area_name'].apply(lambda s: ('%s' % s).encode("utf-8"))
coupons_train['ken_name'] = coupons_train['ken_name'].apply(lambda s: ('%s' % s).encode("utf-8"))
coupons_train['small_area_name'] = coupons_train['small_area_name'].apply(lambda s: ('%s' % s).encode("utf-8"))

print("Coupons train columns")
print(coupons_train.columns.to_series().groupby(coupons_train.dtypes).groups)
print(coupons_train.shape)

coupons_test = pd.read_csv('../input/coupon_list_test.csv', parse_dates=['DISPFROM','DISPEND','VALIDFROM','VALIDEND'])
coupons_test['CAPSULE_TEXT'] = coupons_test['CAPSULE_TEXT'].apply(lambda s: ('%s' % s).encode("utf-8"))
coupons_test['GENRE_NAME'] = coupons_test['GENRE_NAME'].apply(lambda s: ('%s' % s).encode("utf-8"))
coupons_test['large_area_name'] = coupons_test['large_area_name'].apply(lambda s: ('%s' % s).encode("utf-8"))
coupons_test['ken_name'] = coupons_test['ken_name'].apply(lambda s: ('%s' % s).encode("utf-8"))
coupons_test['small_area_name'] = coupons_test['small_area_name'].apply(lambda s: ('%s' % s).encode("utf-8"))

coupons_area_train = pd.read_csv('../input/coupon_area_train.csv')
coupons_area_train['PREF_NAME'] = coupons_area_train['PREF_NAME'].apply(lambda s: ('%s' % s).encode("utf-8"))
coupons_area_train['SMALL_AREA_NAME'] = coupons_area_train['SMALL_AREA_NAME'].apply(lambda s: ('%s' % s).encode("utf-8"))

print("Coupons area train columns")
print(coupons_area_train.columns.to_series().groupby(coupons_area_train.dtypes).groups)
print(coupons_area_train.shape)

coupons_area_test = pd.read_csv('../input/coupon_area_test.csv')
coupons_area_test['PREF_NAME'] = coupons_area_test['PREF_NAME'].apply(lambda s: ('%s' % s).encode("utf-8"))
coupons_area_test['SMALL_AREA_NAME'] = coupons_area_test['SMALL_AREA_NAME'].apply(lambda s: ('%s' % s).encode("utf-8"))


#actual coupon purchases for train data
coupons_details_train = pd.read_csv('../input/coupon_detail_train.csv', parse_dates=['I_DATE'])
coupons_details_train['SMALL_AREA_NAME'] = coupons_details_train['SMALL_AREA_NAME'].apply(lambda s: ('%s' % s).encode("utf-8"))

coupons_train = pd.merge(coupons_train, coupons_details_train , on ='COUPON_ID_hash')


#merge coupon area with coupon data
coupons_train = pd.merge(coupons_train, coupons_area_train, on ='COUPON_ID_hash')
coupons_test = pd.merge(coupons_test, coupons_area_test, on ='COUPON_ID_hash')



#engineer some features 
coupon_df = coupons_train
print(coupon_df.shape)

coupon_df = pd.DataFrame.drop_duplicates(coupon_df)
print(coupon_df.shape)

print(coupons_train.shape)
print(coupons_train['COUPON_ID_hash'].unique().shape)

#merge all data
coupons_train = pd.merge(users, coupons_train, on ='USER_ID_hash')


print(coupons_train.shape)
print(coupons_train.columns)

coupons_train['buy_time_disp_diff'] = coupons_train.I_DATE - coupons_train.DISPFROM

print(coupons_train[1:5])

coupons_train = coupons_train(coupons_train.I_DATE < coupons_train.DISPFROM)
print(coupons_train.shape)

user_ids = users['USER_ID_hash'].tolist()
coupon_ids = coupon_df['COUPON_ID_hash'].tolist()
data = {}

for user_id in user_ids:
    user = users[users['USER_ID_hash']== user_id]
    user_pref_name = user['PREF_NAME']
    #coupons bought by the user
    coupons_bought = coupons_details_train[coupons_details_train['USER_ID_hash']==user_id]
    for coupon_id in coupon_ids:
        coupon = pd.DataFrame.drop_duplicates(coupon_df[coupon_df['COUPON_ID_hash'] == coupon_id])
        key = user_id+'$'+coupon_id
        data[key] = {}
        
        #------ bought coupons of the same genre before ------#
        coupon_disp_from = coupon['DISPFROM'].dt.date
        coupon_genre = coupon['GENRE_NAME']
        if coupon_disp_from.shape[0] != 1 or coupon_genre.shape[0] != 1 :
            print(coupon_disp_from.shape)
            print(coupon_genre.shape)
            print(coupon)
        else:    
            #coupons bought before dipsponibility
            print (coupons_bought['I_DATE'].dt.date)
            coupons_bought_before = coupons_bought[coupons_bought['I_DATE'].dt.date < coupon_disp_from]
            data[key]['genre_count']= coupons_bought_before[coupons_bought_before['GENRE_NAME']==str(coupon_genre)].shape[0]
            
        
        #--------- coupon pref name -----#
        coupon_pref_name = coupon['PREF_NAME']
print(pd.DataFrame.from_dict(data))
        
    






