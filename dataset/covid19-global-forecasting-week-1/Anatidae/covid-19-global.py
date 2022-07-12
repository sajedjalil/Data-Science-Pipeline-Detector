# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
df_submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

inner_list = []
test_new_dic = {}
last_date_test = df_test['Date'][len(df_test) - 1]
country_num = 0
for i in range(len(df_test)):
    forecast_id = df_test['ForecastId'][i]
    province_state = df_test['Province/State'][i]
    country_region = df_test['Country/Region'][i]
    lat = df_test['Lat'][i]
    long = df_test['Long'][i]
    date = df_test['Date'][i]
    inner_dic = {'ForecastId':forecast_id,
                 'Province/State':province_state,
                 'Country/Region':country_region,
                 'Lat':lat,
                 'Long':long,
                 'Date':date
                }
    inner_list.append(inner_dic)
    if date == last_date_test:
        name = str(lat) + '_' + str(long) + '_' + country_region
        test_new_dic[name] = inner_list
        inner_list = []

work_list = []
last_date_train = df_train['Date'][len(df_train) - 1]
first_date_test = df_test['Date'][0]
date_in_status = 0
country_num = 0
for i in range(len(df_train)):
    province_state_train = df_train['Province/State'][i]
    country_region_train = df_train['Country/Region'][i]
    lat_train = df_train['Lat'][i]
    long_train = df_train['Long'][i]
    date_train = df_train['Date'][i]
    confirmed_cases_train = df_train['ConfirmedCases'][i]
    fatalities_train = df_train['Fatalities'][i]
    if date_train == first_date_test:
        date_in_status = 1
    if date_in_status == 1:
        name = str(lat_train) + '_' + str(long_train) + '_' + country_region_train
        target_test_list = test_new_dic[name]
        inner_list = []
        for j in range(len(target_test_list)):
            forecast_id = target_test_list[j]['ForecastId']
            province_state_test = target_test_list[j]['Province/State']
            country_region_test = target_test_list[j]['Country/Region']
            lat_test = target_test_list[j]['Lat']
            long_test = target_test_list[j]['Long']
            date_test = target_test_list[j]['Date']
            if lat_train == lat_test and long_train == long_test and date_train == date_test:
                inner_dic = {'ForecastId':forecast_id,
                             'Province/State':province_state_train,
                             'Country/Region':country_region_train,
                             'Lat':lat_train,
                             'Long':long_train,
                             'Date':date_train,
                             'ConfirmedCases':confirmed_cases_train,
                             'Fatalities':fatalities_train
                            }
                inner_list.append(inner_dic)
        sub_dic = {'country_num':country_num,'list':inner_list}
        work_list.append(sub_dic)
        if date_train == last_date_train:
            country_num += 1
            date_in_status = 0

temp_confirmed_list = []
confirmed_list = []
temp_fatalities_list = []
fatalities_list = []
for i in range(len(df_train)):
    if i != len(df_train) - 1:
        lat = df_train['Lat'][i]
        long = df_train['Long'][i]
        country_region = df_train['Country/Region'][i]
        next_lat = df_train['Lat'][i + 1]
        next_long = df_train['Long'][i + 1]
        next_country_region = df_train['Country/Region'][i + 1]
        if lat == next_lat and long == next_long and country_region == next_country_region:
            inner_confirmed_dic = {'Date':df_train['Date'][i],
                                    'ConfirmedCases':df_train['ConfirmedCases'][i]
                                   }
            temp_confirmed_list.append(inner_confirmed_dic)
            inner_fatalities_dic = {'Date':df_train['Date'][i],
                                     'Fatalities':df_train['Fatalities'][i]
                                    }
            temp_fatalities_list.append(inner_fatalities_dic)
        else:
            confirmed_dic = {'confirmed_list':temp_confirmed_list}
            confirmed_list.append(confirmed_dic)
            temp_confirmed_list = []
            fatalities_dic = {'fatalities_list':temp_fatalities_list}
            fatalities_list.append(fatalities_dic)
            temp_fatalities_list = []
    else:
        confirmed_dic = {'confirmed_list':temp_confirmed_list}
        confirmed_list.append(confirmed_dic)
        temp_confirmed_list = []
        fatalities_dic = {'fatalities_list':temp_fatalities_list}
        fatalities_list.append(fatalities_dic)
        temp_fatalities_list = []

confirmed_list = pd.DataFrame(confirmed_list)
fatalities_list = pd.DataFrame(fatalities_list)

import matplotlib.pyplot as plt
import datetime


predict_days = 60 # MAX

last_date_train = df_train['Date'][len(df_train) - 1]
inner_date_list = last_date_train.split('-')
year = int(inner_date_list[0])
month = int(inner_date_list[1])
day = int(inner_date_list[2])
dt_last = datetime.date(year, month, day)
last_date_submission = df_test['Date'][len(df_test) - 1]
add_date_list = []
predict_list_confirmed_all = []
predict_list_fatalities_all = []
for i in range(predict_days):
    new_date = dt_last + datetime.timedelta(days = i + 1)
    y, m, d = new_date.year, new_date.month, new_date.day
    if m < 10:
        m = '0' + str(m)
    if d < 10:
        d = '0' + str(d)
    new_date = str(y)+'-'+str(m)+'-'+str(d)
    add_date_list.append(new_date)
    if new_date == last_date_submission:
        break
add_date_list = np.array(add_date_list)

confirmed_list = pd.DataFrame(confirmed_list)
fatalities_list = pd.DataFrame(fatalities_list)
for i in range(len(confirmed_list)):
    target_confirmed_list = confirmed_list['confirmed_list'][i]
    df_train_confirmed = pd.DataFrame(target_confirmed_list)
    target_fatalities_list = fatalities_list['fatalities_list'][i]
    df_train_fatalities = pd.DataFrame(target_fatalities_list)

    x = np.array(df_train_confirmed['Date'])
    y_confirmed = np.array(df_train_confirmed['ConfirmedCases'])
    y_fatalities = np.array(df_train_fatalities['Fatalities'])
    x1 = np.arange(len(x))
    #**********************************************************
    dimension = 9
       
    fit_confirmed = np.polyfit(x1, y_confirmed, dimension)
    fit_fatalities = np.polyfit(x1, y_fatalities, dimension+1)
    #**********************************************************
    y2_confirmed = np.poly1d(fit_confirmed)(x1)
    y2_fatalities = np.poly1d(fit_fatalities)(x1)

    # predict
    temp_date = np.append(x, add_date_list)
    predict_list_confirmed = []
    predict_list_fatalities = []
    for j in range(len(x) - 1, len(temp_date)):
        if j == len(x) - 1:
            saved_predict_confirmed = 0
            saved_predict_fatalities = 0
        predict_confirmed = np.poly1d(fit_confirmed)(j)
        predict_confirmed = int(predict_confirmed)
        predict_fatalities = np.poly1d(fit_fatalities)(j)
        predict_fatalities = int(predict_fatalities)
        if predict_confirmed > saved_predict_confirmed:
            predict_list_confirmed.append(predict_confirmed)
            saved_predict_confirmed = predict_confirmed
        else:
            predict_list_confirmed.append(saved_predict_confirmed)
        if predict_fatalities > saved_predict_fatalities:
            predict_list_fatalities.append(predict_fatalities)
            saved_predict_fatalities = predict_fatalities
        else:
            predict_list_fatalities.append(saved_predict_fatalities)
    
    
    predict_list_confirmed_all.append(np.array(predict_list_confirmed))
    predict_list_fatalities_all.append(np.array(predict_list_fatalities))

my_submission_list = []
test_list = []
country_num = 0
for i in range(len(work_list)):
    target_work_list = work_list[i]['list'][0]
    if i == len(work_list) - 1:
        lat_long_train_next = ''
    else:
        next_work_list = work_list[i + 1]['list'][0]
        lat_train_next = next_work_list['Lat']
        long_train_next = next_work_list['Long']
        lat_long_train_next = str(lat_train_next) + '_' + str(long_train_next)
    forecast_id = target_work_list['ForecastId']
    province_state = target_work_list['Province/State']
    country_region = target_work_list['Country/Region']
    lat_train = target_work_list['Lat']
    long_train = target_work_list['Long']
    lat_long_train = str(lat_train) + '_' + str(long_train)
    if lat_long_train != lat_long_train_next:
        country_num += 1
    date = target_work_list['Date']
    confirmed_cases = target_work_list['ConfirmedCases']
    fatalities = target_work_list['Fatalities']
    inner_dic = {'ForecastId':forecast_id,
                 'ConfirmedCases':confirmed_cases,
                 'Fatalities':fatalities
                }
    test_dic = {'ForecastId':forecast_id,
                 'ConfirmedCases':confirmed_cases,
                 'Fatalities':fatalities,
                 'Date':date
                }
    my_submission_list.append(inner_dic)
    test_list.append(test_dic)
    
    if date == last_date_train:
        target_confirmed_list = predict_list_confirmed_all[country_num]
        target_fatalities_list = predict_list_fatalities_all[country_num]
        for j in range(len(add_date_list)):
            forecast_id += 1
            confirmed_cases = target_confirmed_list[j]
            fatalities = target_fatalities_list[j]
            date = add_date_list[j]
            inner_dic = {'ForecastId':forecast_id,
                         'ConfirmedCases':confirmed_cases,
                         'Fatalities':fatalities
                        }
            test_dic = {'ForecastId':forecast_id,
                         'ConfirmedCases':confirmed_cases,
                         'Fatalities':fatalities,
                         'Date':date
                        }
            my_submission_list.append(inner_dic)
            test_list.append(test_dic)

my_submission_list = pd.DataFrame(my_submission_list)
my_submission_list.to_csv(path_or_buf='submission.csv', index=False)