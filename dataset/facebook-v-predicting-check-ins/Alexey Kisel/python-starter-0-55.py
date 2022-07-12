import pandas as pd
import numpy as np
import datetime
import time
import os
import math
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

start_time = time.time()

size = 10.0;

#TrainFields = ['x','y','accuracy', 'hour', 'weekday', 'month', 'year','x_t_y', 'quarter_hour']
TrainFields = ['x','y','accuracy', 'hour', 'weekday', 'month', 'day', 'year','x_t_y']
TH = 5
x_step = 0.3
y_step = 0.15
border_augment_y = 0.01
border_augment_x = 0.03

x_ranges = zip(np.arange(0, size, x_step), np.arange(x_step, size + x_step, x_step));
y_ranges = zip(np.arange(0, size, y_step), np.arange(y_step, size + y_step, y_step));

print('Calculate hour, weekday, month and year for train and test')

initial_date = np.datetime64('2014-01-01T01:01', dtype='datetime64[m]')

train_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') for mn in train.time.values)  
test_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') for mn in test.time.values)  

train['hour'] = train_times.hour 
train['weekday'] = train_times.weekday 
train['day'] = (train_times.dayofyear).astype(int)
train['month'] = train_times.month
train['year'] = (train_times.year - 2013).astype(int)
train['x_t_y'] = train['x'] * train['y']

test['hour'] = test_times.hour 
test['weekday'] = test_times.weekday 
test['day'] = (test_times.dayofyear).astype(int)
test['month'] = test_times.month
test['year'] = (test_times.year - 2013).astype(int)
test['x_t_y'] = test['x'] * test['y']

# cols = ['hour',  'weekday',  'month']

# for cl in cols:
#   ave = train[cl].mean()
#   std = train[cl].std()
#   train[cl] = (train[cl].values - ave ) / std

# for cl in cols:
#   ave = test[cl].mean()
#   std = test[cl].std()
#   test[cl] = (test[cl].values - ave ) / std

# X_train = train[TrainFields];
# y_train = train[['place_id']];
# X_test = test[TrainFields];
# X_test_labels = test[['row_id']];

preds_total = pd.DataFrame();
for x_min, x_max in  x_ranges:
    start_time_row = time.time()
    for y_min, y_max in  y_ranges:
        x_max = round(x_max, 4)
        x_min = round(x_min, 4)
        
        y_max = round(y_max, 4)
        y_min = round(y_min, 4)
        
        if x_max == size:
          x_max = x_max + 0.001
            
        if y_max == size:
          y_max = y_max + 0.001
            
        train_grid = train[(train['x'] >= x_min - border_augment_x) &
                           (train['x'] < x_max + border_augment_x) &
                           (train['y'] >= y_min - border_augment_y) &
                           (train['y'] < y_max + border_augment_y)]

        test_grid = test[(test['x'] >= x_min) &
                         (test['x'] < x_max) &
                         (test['y'] >= y_min) &
                         (test['y'] < y_max)]

        place_counts = train_grid.place_id.value_counts()
        mask = place_counts[train_grid.place_id.values] >= TH
        train_grid = train_grid.loc[mask.values]

        X_train_grid = train_grid[TrainFields];
        y_train_grid = train_grid[['place_id']].values.ravel();
        X_test_grid = test_grid[TrainFields];
        
        clf = RandomForestClassifier(n_estimators = 1000, n_jobs = -1,random_state=0);
        clf.fit(X_train_grid, y_train_grid)        
        
        preds = dict(zip([el for el in clf.classes_], zip(*clf.predict_proba(X_test_grid))))
        preds = pd.DataFrame.from_dict(preds)
        
        preds['0_'], preds['1_'], preds['2_'] = zip(*preds.apply(lambda x: preds.columns[x.argsort()[::-1][:3]].tolist(), axis=1));
        preds = preds[['0_','1_','2_']];
        preds['row_id'] = test_grid['row_id'].reset_index(drop=True);
        preds_total = pd.concat([preds_total, preds], axis=0);
    print("Elapsed time row: %s minutes" % ( (time.time() - start_time_row ) / 60.0 ))

preds_total['place_id'] = preds_total[['0_', '1_', '2_']].apply(lambda x: ' '.join([str(x1) for x1 in x]), axis=1)
preds_total.drop('0_', axis=1, inplace=True)
preds_total.drop('1_', axis=1, inplace=True)
preds_total.drop('2_', axis=1, inplace=True)
sub_file = os.path.join('rf_submission_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
preds_total.to_csv(sub_file,index=False)
print("Elapsed time overall: %s minutes" % ( (time.time() - start_time) / 60.0))