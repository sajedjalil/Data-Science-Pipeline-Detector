#
# A top scoring model that uses
# three (recursive) exponential equations.
# Manually specified - not ML.
#

import numpy as np
import csv

from datetime import date, timedelta

fname_train ='/kaggle/input/covid19-global-forecasting-week-5/train.csv'
fname_test ='/kaggle/input/covid19-global-forecasting-week-5/test.csv'
fname_submit = 'submission.csv'

#
# DATA -- everything in one place
#

class Data():

    def __init__(self):
        self.data_train = {}
        self.data_pred = {}
    
    def load_actuals(self, filename, final_train_date):
        with open(fname_train) as f_train:
            for fields in csv.DictReader(f_train):
                key = self.make_key(fields)
                if fields['Date'] <= final_train_date:
                    self.data_train[key] = fields['TargetValue']
    
     
    # concatenate date, location, target, to uniquely
    # identify a target value in dictionary
    def make_key(self, fields, asof_date=None):
        key = fields['Date'] if asof_date is None else asof_date
        for k in ['County', 'Province_State', 'Country_Region', 'Target']:
            key += fields[k]
        return key

    
    # return list with values from range of days
    # eg, for today and the 6 days before
    # uses data_pred if key not found in data_train 
    def get_window(self, fields, start, stop, start_date=None):
        start_date = date.fromisoformat(fields['Date']) if start_date is None else date.fromisoformat(start_date)
        window = []
        for i in range(start, stop):
            d = start_date + timedelta(days=i)
            k = self.make_key(fields, d.isoformat())
            v  = float(
                self.data_train[k] if k in self.data_train 
                else self.data_pred[k])    
            window.append(v)
        return window

    # remember a predicted value
    def add_prediction(self, fields, value):
        k = self.make_key(fields)
        self.data_pred[k] = value

#
# MODEL -- three groups: rising, falling, all others
# 

def predict_three(data, fields):
    
    # six day window: -8, -7, -6, -5, -4, -3
    window = data.get_window(fields, -8, -2)

    loc = fields['Country_Region'] + fields['Province_State'] + fields['County']

    rising = (
        'Brazil',
        'Mexico',
        'Chile',
        'India',
        'Peru',
        'South Africa',
        'Egypt'
        )

    falling = (
        'USNew York', 
        'USNew YorkNew York',
        'USNew YorkKings',
        'USNew Jersey',
        'USNew JerseyBergen',
        'USNew JerseyEssex',
        'France','Italy','Spain','United Kingdom','Belgium'
        )

    # today will be like last week, more or less
    if loc in rising:
        p50 = 1.2 * np.mean(window) 
    elif loc in falling:
        p50 = 0.7 * np.mean(window) 
    else:
        p50 = 0.9 * np.mean(window) 

    # quantiles based on prediction
    if p50 < 25:
        p05 = 0
        p95 = 3.0 * p50
    else:
        p05 = 0.5 * p50
        p95 = 1.7 * p50

    return p05, p50, p95

#
# RUN
#

# LEAKAGE PREVENTION
# ignore data after 5/9 to ensure predictions
# are comparable to actual submissions
FINAL_TRAIN_DATE = '2020-05-09'

data = Data()
data.load_actuals(fname_train, FINAL_TRAIN_DATE)

with open(fname_test) as f_test:
    with open(fname_submit, mode='w') as f_submit:
        f_submit.write('ForecastId_Quantile,TargetValue\n')
        for fields in csv.DictReader(f_test):

            p05, p50, p95 = predict_three(data, fields) 
            data.add_prediction(fields, p50)

            id = fields['ForecastId']
            f_submit.write(f'{id}_0.05, {p05}\n')
            f_submit.write(f'{id}_0.5, {p50}\n')
            f_submit.write(f'{id}_0.95, {p95}\n')



