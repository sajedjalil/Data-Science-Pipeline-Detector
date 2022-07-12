# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import xgboost as xgb
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

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

size = 10.0;

x_step = 0.5
y_step = 0.25

x_ranges = zip(np.arange(0, size, x_step), np.arange(x_step, size + x_step, x_step));
y_ranges = zip(np.arange(0, size, y_step), np.arange(y_step, size + y_step, y_step));

def prepare_data(df):
    """
    Feature engineering and computation of the grid.
    """
    #Feature engineering
    fw = [500, 1000, 4, 3, 1./22., 2, 10] #feature weights (black magic here)
    df.x = df.x.values * fw[0]
    df.y = df.y.values * fw[1]
    initial_date = np.datetime64('2014-01-01T01:01', dtype='datetime64[m]') 
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') 
                               for mn in df.time.values)    
    df['hour'] = d_times.hour * fw[2]
    df['weekday'] = d_times.weekday * fw[3]
    df['day'] = (d_times.dayofyear * fw[4]).astype(int)
    df['month'] = d_times.month * fw[5]
    df['year'] = (d_times.year - 2013) * fw[6]

    return df

def model(x_ranges, y_ranges, x_end, y_end, train, test, clf, raw_output, th):   
    start_time = time.time()
    preds_total = pd.DataFrame();
    for x_min, x_max in  x_ranges:
        start_time_row = time.time()
        for y_min, y_max in  y_ranges: 
            start_time_cell = time.time()
            x_max = round(x_max, 4)
            x_min = round(x_min, 4)

            y_max = round(y_max, 4)
            y_min = round(y_min, 4)

            if x_max == x_end:
                x_max = x_max + 0.001

            if y_max == y_end:
                y_max = y_max + 0.001

            train_grid = train[(train['x'] >= x_min - 0.15) &
                               (train['x'] < x_max + 0.15) &
                               (train['y'] >= y_min - 0.075) &
                               (train['y'] < y_max + 0.075)]
            
            test_grid = test[(test['x'] >= x_min) &
                             (test['x'] < x_max) &
                             (test['y'] >= y_min) &
                             (test['y'] < y_max)]
            
            train_grid = prepare_data(train_grid);
            test_grid = prepare_data(test_grid);
            
            if th > 0:
                train_grid = train_grid.groupby("place_id").filter(lambda x: len(x) >= th)

            X_train_grid = train_grid[['x','y',
                                       #'accuracy',
                                       'hour',
                                       'day',
                                       'year',
                                       'weekday',
                                       'month']];
            
            y_train_grid = train_grid[['place_id']].values.ravel();
            
            X_test_grid = test_grid[['x','y',
                                     #'accuracy', 
                                     'hour',
                                     'day',
                                     'year',
                                     'weekday',
                                     'month']];
            
            #Preparing data
            le = LabelEncoder()
            y = le.fit_transform(y_train_grid)

            clf.fit(X_train_grid, y)
            y_pred = clf.predict_proba(X_test_grid)

            preds = le.inverse_transform(np.argsort(y_pred, axis=1)[:,::-1][:,:3])
            preds = pd.DataFrame.from_dict(preds)
            preds['row_id'] = test_grid['row_id'].reset_index(drop=True);
            preds_total = pd.concat([preds_total, preds], axis=0);
        print("Elapsed time row: %s minutes" % ((time.time() - start_time_row)/60))
    print("Elapsed time overall: %s minutes" % ((time.time() - start_time)/60))
    
    print(preds_total.shape)

    preds_total = preds_total.sort_values(by='row_id', axis=0, ascending=True);
    return preds_total;


clf = KNeighborsClassifier(n_neighbors=25, weights='distance', 
                                       metric='manhattan', n_jobs=-1)

preds_total = model(x_ranges, y_ranges, size, size, train, test, clf, 'rf_1200_020_008/', 5)
preds_total.columns = ['l1', 'l2', 'l3', 'row_id'];
preds_total['place_id'] = preds_total['l1'].apply(str) + ' ' + preds_total['l2'].apply(str) + ' ' + preds_total['l3'].apply(str);
preds_total[['row_id','place_id']].to_csv('sub_knn_overlap_.csv', index = False);