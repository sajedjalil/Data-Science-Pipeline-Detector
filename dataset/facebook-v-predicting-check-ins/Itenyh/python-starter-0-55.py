import pandas as pd
import numpy as np
import datetime
import time
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
start_time = time.time()

size = 10.0;

x_step = 0.2
y_step = 0.2

x_ranges = zip(np.arange(0, size, x_step), np.arange(x_step, size + x_step, x_step));
y_ranges = zip(np.arange(0, size, y_step), np.arange(y_step, size + y_step, y_step));
#print x_ranges
#print y_ranges

print('Calculate hour, weekday, month and year for train and test')
train['hour'] = (train['time']//60)%24+1 # 1 to 24
train['weekday'] = (train['time']//1440)%7+1
train['month'] = (train['time']//43200)%12+1 # rough estimate, month = 30 days
train['year'] = (train['time']//525600)+1 

test['hour'] = (test['time']//60)%24+1 # 1 to 24
test['weekday'] = (test['time']//1440)%7+1
test['month'] = (test['time']//43200)%12+1 # rough estimate, month = 30 days
test['year'] = (test['time']//525600)+1

#print 'shape after time engineering'
#print train.shape
#print test.shape

X_train = train[['x','y','accuracy','time', 'hour', 'weekday', 'month', 'year']];
y_train = train[['place_id']];
X_test = test[['x','y','accuracy','time', 'hour', 'weekday', 'month', 'year']];
X_test_labels = test[['row_id']];

preds_total = pd.DataFrame();
for x_min, x_max in  x_ranges:
    start_time_row = time.time()
    for y_min, y_max in  y_ranges: 
        start_time_cell = time.time()
        x_max = round(x_max, 4)
        x_min = round(x_min, 4)
        
        y_max = round(y_max, 4)
        y_min = round(y_min, 4)
        
        if x_max == size:
            x_max = x_max + 0.001
            
        if y_max == size:
            y_max = y_max + 0.001
            
        train_grid = train[(train['x'] >= x_min) &
                           (train['x'] < x_max) &
                           (train['y'] >= y_min) &
                           (train['y'] < y_max)]

        test_grid = test[(test['x'] >= x_min) &
                         (test['x'] < x_max) &
                         (test['y'] >= y_min) &
                         (test['y'] < y_max)]
        
        X_train_grid = train_grid[['x','y','accuracy', 'hour', 'weekday', 'month', 'year']];
        y_train_grid = train_grid[['place_id']].values.ravel();
        X_test_grid = test_grid[['x','y','accuracy','hour', 'weekday', 'month', 'year']];
        
        #clf = GradientBoostingClassifier();
        #clf =  LogisticRegression(multi_class='multinomial', solver = 'lbfgs');
        #clf = xgb.XGBClassifier(n_estimators=10);
        clf = RandomForestClassifier(n_estimators = 10, n_jobs = -1,random_state=0);
        clf.fit(X_train_grid, y_train_grid)
        
        preds = dict(zip([el for el in clf.classes_], zip(*clf.predict_proba(X_test_grid))))
        preds = pd.DataFrame.from_dict(preds)
        
        preds['0_'], preds['1_'], preds['2_'] = zip(*preds.apply(lambda x: preds.columns[x.argsort()[::-1][:3]].tolist(), axis=1));
        preds = preds[['0_','1_','2_']];
        preds['row_id'] = test_grid['row_id'].reset_index(drop=True);
        preds_total = pd.concat([preds_total, preds], axis=0);
        print("Elapsed time cell: %s seconds" % (time.time() - start_time_cell))
    print("Elapsed time row: %s seconds" % (time.time() - start_time_row))

preds_total['place_id'] = preds_total[['0_', '1_', '2_']].apply(lambda x: ' '.join([str(x1) for x1 in x]), axis=1)
preds_total.drop('0_', axis=1, inplace=True)
preds_total.drop('1_', axis=1, inplace=True)
preds_total.drop('2_', axis=1, inplace=True)
sub_file = os.path.join('rf_submission_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
preds_total.to_csv(sub_file,index=False)
print("Elapsed time overall: %s seconds" % (time.time() - start_time))