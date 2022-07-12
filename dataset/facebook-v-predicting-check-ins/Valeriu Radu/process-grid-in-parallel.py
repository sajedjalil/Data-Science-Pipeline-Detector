# coding: utf-8
__author__ = 'Valeriu Radu: https://www.kaggle.com/valeriur'

import multiprocessing
from multiprocessing import Process
from multiprocessing import Manager

import math
import xgboost
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import datetime
import time
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from ml_metrics import mapk
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

def prepare_data(df):
    """
    Feature engineering
    """
    minute = df.time % 60
    df['hour'] = df['time'] // 60
    #df.drop(['time'], axis=1, inplace=True)
    df['weekday'] = df['hour'] // 24
    df['month'] = df['weekday'] // 30
    df['year'] = (df['weekday'] // 365 + 1) * 10.0
    df['hour'] = ((df['hour'] % 24 + 1) + minute / 60.0) * 4.0
    df['weekday'] = (df['weekday'] % 7 + 1) * 3.0
    df['month'] = (df['month'] % 12 + 1) * 2.0
    df['accuracy'] = np.log10(df['accuracy']) * 10.0

    return df

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train = prepare_data(train)
test = prepare_data(test)

print(train.shape)
print(test.shape)

def calculate_distance(distances):
    return distances ** -2

def process_column(x_min, x_max, y_ranges, x_end, y_end, train, test, preds_total):
    start_time_column = time.time()
    preds_total[x_min] = pd.DataFrame();
    for y_min, y_max in  y_ranges:

            start_time_cell = time.time()

            if x_max == x_end:
                x_max = x_max + 0.001

            if y_max == y_end:
                y_max = y_max + 0.001

            train_cell = train[(train['x'] >= x_min - 0.03) &
                               (train['x'] < x_max + 0.03) &
                               (train['y'] >= y_min - 0.015) &
                               (train['y'] < y_max + 0.015)]

            add_data = train_cell[train_cell.hour<10]# add data for periodic time that hit the boundary
            add_data.hour = add_data.hour+96
            train_cell = train_cell.append(add_data)
            add_data = train_cell[train_cell.hour>90]
            add_data.hour = add_data.hour-96
            train_cell = train_cell.append(add_data)
            del add_data

            train_cell = train_cell.drop(['time'], axis=1)
            train_cell = train_cell.groupby("place_id").filter(lambda x: len(x) >= 8)

            test_cell = test[(test['x'] >= x_min) &
                             (test['x'] < x_max) &
                             (test['y'] >= y_min) &
                             (test['y'] < y_max)]

            row_ids = test_cell['row_id'].reset_index(drop=True);
            test_cell = test_cell.drop(['row_id', 'time'], axis=1)

            #Feature engineering on x and y
            train_cell.loc[:,'x'] *= 490.0
            train_cell.loc[:,'y'] *= 980.0
            test_cell.loc[:,'x'] *= 490.0
            test_cell.loc[:,'y'] *= 980.0

            le = LabelEncoder()

            y = le.fit_transform(train_cell.place_id.values)
            X = train_cell.drop(['row_id', 'place_id'], axis=1)

            #Applying the classifier
            clf = KNeighborsClassifier(n_neighbors=np.floor(np.sqrt(y.size)/5.2632).astype(int),
                                    weights=calculate_distance, metric='manhattan',n_jobs=-1)

            clf.fit(X, y)

            y_pred = clf.predict_proba(test_cell.values)

            preds = le.inverse_transform(np.argsort(y_pred, axis=1)[:,::-1][:,:3])
            preds = pd.DataFrame.from_dict(preds)
            preds['row_id'] = row_ids;
            preds = preds.set_index('row_id')
            preds.index.name = 'row_id';
            preds_total[x_min] = pd.concat([preds_total[x_min], preds], axis=0);

    print("Elapsed time column: %s minutes" % ((time.time() - start_time_column)/60))

def model(x_ranges, y_ranges, x_end, y_end, train, test):
    start_time = time.time()
    jobs = []
    mgr = Manager()
    preds_total = mgr.dict();

    for x_min, x_max in  x_ranges:
        p = multiprocessing.Process(target=process_column, args=(x_min, x_max, y_ranges, \
                                                                 x_end, y_end, train, test, preds_total))
        jobs.append(p)
        p.start()
        if len(jobs) == 2:
            for proc in jobs:
                proc.join();
            jobs = [];

    print("Elapsed time overall: %s minutes" % ((time.time() - start_time)/60))

    preds_total = pd.concat(preds_total.values(), axis=0);
    print(preds_total.shape)

    return preds_total.sort_index();

def xfrange(start, end, step):
    gens = [];
    end = round(end, 2)
    start = round(start, 2)
    while(start < end):
        gens.append(start)
        start = round(start + step, 2)

    return gens

def gen_ranges(start, end, step):
    return zip(xfrange(start, end, step), xfrange(start + step, end + step, step));

size = 10.0;

x_step = 0.5
y_step = 0.25

x_ranges = gen_ranges(0, size, x_step);
y_ranges = gen_ranges(0, size, y_step);


preds_total = model(x_ranges, y_ranges, size, size, train, test)
preds_total = preds_total.applymap(str)
preds_total.columns = ['l1', 'l2', 'l3'];
print('Writing submission file')
preds_total = preds_total.l1.str.cat([preds_total.l2, preds_total.l3], sep=' ')
preds_total.columns = ['place_id'];
preds_total.to_csv('submission_sample_parallel.csv', index=True, header=True, index_label='row_id')