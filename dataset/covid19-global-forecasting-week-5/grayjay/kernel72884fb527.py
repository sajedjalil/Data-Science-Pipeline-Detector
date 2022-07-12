# simple autoregressive model

import numpy as np
import os.path
import csv

from datetime import date, timedelta


fname_train ='/kaggle/input/covid19-global-forecasting-week-5/train.csv'
fname_test ='/kaggle/input/covid19-global-forecasting-week-5/test.csv'
fname_submit = 'submission.csv'

# our two global dictionaries
data_train = {}
data_pred = {}

# concatenate location, target, date to uniquely
# identify a target value in dictionary
def make_key(fields, date=None):
    key = fields['Date'] if date is None else date.isoformat()
    for k in ['County', 'Province_State', 'Country_Region', 'Target']:
        key += fields[k]
    return key

# return list with values from range of days
# eg, for today and the 6 days before
# uses data_pred if date not found in data_train 
def get_window(fields, start, stop):
    start_date = date.fromisoformat(fields['Date'])
    window = []
    for i in range(start, stop):
        d = start_date + timedelta(days=i)
        k = make_key(fields, d)
        v  = float(
            data_train[k] if k in data_train 
            else data_pred[k])
        
        window.append(v)
    return window

# predict quantile values for a row in the test file
def predict(fields):
    
    # predictions based on last week, more or less
    window = get_window(fields, -8, -2)
    if fields['Target'] == 'Fatalities':
        p05 = 0.6 * min(window)
        p50 = 0.8 * np.mean(window[0:5]) 
        p95 = 1.1 * max(window)
    else:
        p05 = 0.7 * min(window)
        p50 = 0.9 * np.mean(window[0:6]) 
        p95 = 1.2 * max(window)

    # side effect: remember prediction for future use
    k = make_key(fields)
    data_pred[k] = p50

    return p05, p50, p95

# read all training target values into one big dictionary
with open(fname_train) as f_train:
    for fields in csv.DictReader(f_train):
        key = make_key(fields)
        data_train[key] = fields['TargetValue']

# read test file, predict values, write values
with open(fname_test) as f_test:
    with open(fname_submit, mode='w') as f_submit:
        f_submit.write('ForecastId_Quantile,TargetValue\n')
        for fields in csv.DictReader(f_test):
            p05, p50, p95 = predict(fields) 
            id = fields['ForecastId']
            f_submit.write(f'{id}_0.05, {p05}\n')
            f_submit.write(f'{id}_0.5, {p50}\n')
            f_submit.write(f'{id}_0.95, {p95}\n')

