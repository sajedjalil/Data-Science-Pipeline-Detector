from datetime import datetime
import random
import numpy as np
import pandas as pd
# import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

week_dict = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 1}
dates = pd.date_range("2016-01-01", "2017-05-31", freq='D')
test_dates = pd.date_range("2017-04-23", "2017-05-31", freq='D')

##### history of number of visit #####
path = '../input/'
air_visit = pd.read_csv(path+"air_visit_data.csv", parse_dates=["visit_date"])
air_visit_dcast = air_visit.pivot(index='air_store_id', columns='visit_date', values='visitors')
cols = ["date"] + ["b"+str(i)+"d_visit" for i in range(7*53+1)]  


data_visit = pd.DataFrame()
for d in dates:
    temp = air_visit_dcast.copy()
    temp.rename(columns={i:"b" + str((d-i).days) + "d_visit" for i in temp.columns}, inplace=True)
    temp['date'] = d
    temp = temp.reindex(cols, axis=1)
    data_visit = pd.concat([data_visit, temp], axis=0)

data_visit = data_visit.loc[data_visit.b0d_visit.notnull() | (data_visit.date >= datetime.strptime("2017-04-23", "%Y-%m-%d"))]


##### prepare other data #####
air_reserve = pd.read_csv(path+"air_reserve.csv", parse_dates=["visit_datetime", 'reserve_datetime'])
air_reserve['date']=air_reserve.visit_datetime.dt.date

date_info = pd.read_csv(path+"date_info.csv", parse_dates=["calendar_date"])
date_info['dow'] = date_info.calendar_date.dt.weekday.map(week_dict)
date_info['holiday_flg']=[1 if v in [1,7] else date_info.holiday_flg[i] for i, v in enumerate(date_info['dow'])]
date_info['holiday_flgs3']=pd.concat([date_info.holiday_flg.shift(1).fillna(1),
                                        date_info.holiday_flg,
                                        date_info.holiday_flg.shift(-1).fillna(0)],axis=1).apply(lambda row: ''.join(map(str,map(int,row))), axis=1)  # 节假日前一天-当天-后一天
normal_date = date_info.loc[((date_info.dow==1) & (date_info.holiday_flgs3=="110")) |
                            ((date_info.dow==2) & (date_info.holiday_flgs3=="100")) |
                            ((date_info.dow==3) & (date_info.holiday_flgs3=="000")) |
                            ((date_info.dow==4) & (date_info.holiday_flgs3=="000")) |
                            ((date_info.dow==5) & (date_info.holiday_flgs3=="000")) |
                            ((date_info.dow==6) & (date_info.holiday_flgs3=="001")) |
                            ((date_info.dow==7) & (date_info.holiday_flgs3=="011"))] 


air_store_info = pd.read_csv(path+"air_store_info.csv", usecols=[0,1,3,4])
le = LabelEncoder()
air_store_info['air_genre_name'] = le.fit_transform(air_store_info['air_genre_name'])
for j in range(max(air_store_info.air_genre_name)+1):
    if j in [1-1,6-1,10-1]:
        continue
    air_store_info["genre_"+str(j)] = (air_store_info.air_genre_name == j)*1
air_store_info.drop('air_genre_name', axis=1, inplace=True)


##### make xgboost model and prediction for each day #####
key = ["date", "air_store_id"]
target = "b0d_visit"

prediction = pd.DataFrame()
for day in range(test_dates.shape[0]):  
    print(test_dates[day])
    test_dow = week_dict.get(test_dates[day].weekday())
    day += 1                          

    ##### select dates for train #####
    if day in range(10,13+1):
        train_dates = normal_date.loc[normal_date.dow == test_dow].calendar_date
    else:
        if test_dow == 1:
            train_dates = normal_date.loc[normal_date.dow.isin([1,7])].calendar_date
        elif test_dow == 2:
            train_dates = normal_date.loc[normal_date.dow.isin([2,3,5])].calendar_date
        elif test_dow == 3:
            train_dates = normal_date.loc[normal_date.dow.isin([2,3,4,5])].calendar_date
        elif test_dow == 4:
            train_dates = normal_date.loc[normal_date.dow.isin([3,4,5])].calendar_date
        elif test_dow == 5:
            train_dates = normal_date.loc[normal_date.dow.isin([3,4,5])].calendar_date
        elif test_dow == 6:
            train_dates = normal_date.loc[normal_date.dow.isin([4,6,7])].calendar_date
        else:# test_dow == 7:
            train_dates = normal_date.loc[normal_date.dow.isin([1,6,7])].calendar_date
    t1 = datetime.strptime("2016-01-04",'%Y-%m-%d')
    t2 = datetime.strptime("2016-12-31", '%Y-%m-%d')
    t3 = datetime.strptime("2017-01-04", '%Y-%m-%d')
    t4 = datetime.strptime("2017-04-23", '%Y-%m-%d')
    train_dates = train_dates.loc[((t1 < train_dates) & (train_dates < t2)) | ((t3 < train_dates) & (train_dates < t4))]
    train_dates = train_dates.sort_values(ascending=False)
    train_dates = train_dates.iloc[0:min(150,len(train_dates))] 


    ##### history of number of reserve #####
    cols = ["date"] + ["b"+str(i)+"d_reserve" for i in range(7*10+1)]

    data_reserve = pd.DataFrame()
    for d in train_dates.tolist() + [test_dates[day-1]]:
        d = pd.Timestamp(d, tz=None, freq='D')
        temp = air_reserve.loc[air_reserve.reserve_datetime < (d + 1 - day)].groupby(['air_store_id',
                                                                                      'date'], as_index=False).agg({'reserve_visitors':'sum'})
        if temp.shape[0] == 0:
            continue
        air_reserve_dcast = temp.pivot(index='air_store_id', columns='date', values='reserve_visitors')
        temp = air_reserve_dcast.copy()
        temp.rename(
            columns={i: "b" + str((d - pd.Timestamp(i, tz=None, freq='D')).days) + "d_reserve" for i in temp.columns},
            inplace=True)
        temp['date'] = d
        temp = temp.reindex(cols, axis=1)
        data_reserve = pd.concat([data_reserve, temp], axis=0)


    ##### train data #####
    data_train = data_visit.loc[data_visit.date.isin(train_dates)]
    
    if day == 10:
        # regard 2017-05-02 as Friday
        data_train.b0d_visit=data_train.b4d_visit
    elif day == 11:
        # regard 2017-05-03 as Saturday
        data_train.b0d_visit=data_train.b4d_visit
    elif day == 12:
        # regard 2017-05-04 as Saturday
        data_train.b0d_visit=data_train.b5d_visit
    elif day == 13:
        # regard 2017-05-05 as Saturday
        data_train.b0d_visit=data_train.b6d_visit
    
    data_train = data_train.loc[data_train[target].notnull()]
    data_train = pd.merge(data_train.reset_index(), data_reserve.reset_index(), on=["date","air_store_id"], how='left')


    ##### test data #####
    data_test = data_visit.loc[data_visit.date == test_dates[day-1]]
    data_test = pd.merge(data_test.reset_index(), data_reserve.reset_index(), on=["date","air_store_id"], how='left')


    ##### log transform #####
    for j in range(2,data_train.shape[1]):
        data_train.iloc[:,j] = np.log1p(data_train.iloc[:,j])
        data_test.iloc[:,j] = np.log1p(data_test.iloc[:,j])


    ##### select and add features #####
    f1 = [i * 7 for i in list(range(1,53+1))]
    f2 = [i + day for i in list(range(20+1))]
    feature_days = sorted(set(f1+f2))
    feature_days = [i for i in feature_days if i >= day]
    exp_vars = ["b"+str(i)+"d_visit" for i in feature_days]
    f1 = list(range(7+1))
    f2 = [i * 7 for i in list(range(10+1))]
    feature_days = sorted(set(f1+f2))
    exp_vars.extend(["b"+str(i)+"d_reserve" for i in feature_days])

    for j in range(1,5+1):
        feature_days = [i + (j-1)*7 for i in list(range(day,(day+6)+1))]
        data_train['b' + str(j) + 'w'] = data_train.loc[:, ["b" + str(i) + "d_visit" for i in feature_days]].mean(
            axis=1)
        data_test['b' + str(j) + 'w'] = data_test.loc[:, ["b" + str(i) + "d_visit" for i in feature_days]].mean(
            axis=1)
        exp_vars.extend(["b"+str(j)+"w"])

    feature_days =[(i + int((day-1)/7)) * 7 for i in range(1,8+1)]
    data_train['dow_mean8w'] = data_train.loc[:, ["b" + str(i) + "d_visit" for i in feature_days]].mean(axis=1)
    data_test['dow_mean8w'] = data_test.loc[:, ["b" + str(i) + "d_visit" for i in feature_days]].mean(axis=1)

    exp_vars.extend(["dow_mean8w"])
    exp_vars = [i for i in exp_vars if i in data_train.columns]

    data_train = data_train.loc[:, key + [target] + exp_vars]
    data_test = data_test.loc[:, key + exp_vars]

    data_train = pd.merge(data_train, air_store_info, on="air_store_id")
    data_test = pd.merge(data_test, air_store_info, on="air_store_id")

    temp = data_train.date.dt.weekday
    temp = temp.map(week_dict)
    if len(temp.unique()) <= 2:
        data_train['dow'] = data_train.date.dt.weekday.map(week_dict)
        data_test['dow'] = data_train.date.dt.weekday.map(week_dict)
    else:
        for j in range(1,7+1):
            if sum(temp == j) > 0:
                data_train["dow_"+str(j)] = (temp == j)*1
                data_test["dow_"+str(j)] = (data_test.date.dt.weekday.map(week_dict) == j)*1


    data_train['day'] = data_train.date.dt.day
    data_test['day'] = data_test.date.dt.day

    exp_vars = [i for i in data_train.columns if i not in key + [target]]


    ##### modeling #####
    print("=" * 50)
    print("Step %d" % (day))
    print("=" * 50)
    x_train = data_train.loc[:, exp_vars]
    y_train = data_train[target].values
    dtrain = lgb.Dataset(x_train, label=y_train)

    x_test = data_test.loc[:, exp_vars]
    
    params = {
        "learning_rate": 0.02,
        "max_depth": 8,
        "min_child_weight": 16,  # min_sum_hessian_in_leaf
        "bagging_fraction": 0.9,  # subsample
        "feature_fraction": 0.9,  # colsample_bytree
        'bagging_freq': 2,
        "objective": "regression",
        "metric": "rmse"
    }

    random.seed(0)
    model_bst = lgb.train(params,
                          dtrain,
                          num_boost_round=300 if day in range(10, 13+1) else 700,
                          verbose_eval=100,
                          # num_thread=4
                          )

    print("\n".join(("%s: %.2f" % x) for x in sorted(
        zip(x_train.columns, model_bst.feature_importance("gain")),
        key=lambda x: x[1], reverse=True
    )))
    ##### predict #####
    temp = data_test.loc[:, key]
    temp['visitors'] = model_bst.predict(x_test, num_iteration=model_bst.best_iteration)
    temp['visitors'] = np.expm1(temp['visitors']).clip(0.)
    temp['date'] = temp.date.dt.strftime('%Y-%m-%d')
    temp['id'] = temp.loc[:,['air_store_id','date']].apply(lambda x: '_'.join(map(str,x)), axis=1)
    prediction = pd.concat([prediction, temp.loc[:, ['id', 'visitors']]], axis=0)


##### make submission #####
submission = pd.read_csv(path+"sample_submission.csv", usecols=[0])
submission = pd.merge(submission, prediction, on="id")
submission.loc[submission.visitors < 1, ['visitors']] = 1
submission.to_csv("submission.csv", index=False)
