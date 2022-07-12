import pandas as pd
import numpy as np
import datetime
import time
import os
import random
from heapq import nlargest
from operator import itemgetter
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

pd.options.mode.chained_assignment = None  

# constants -----------------------------------------------
CROSS_VALIDATION = 0.2

WINDOW_LEN       = 0.5
WINDOW_X         = 4.0 #random.random() * (10 - WINDOW_LEN)
WINDOW_Y         = 4.0 #random.random() * (10 - WINDOW_LEN)

STEP_X           = 0.07
STEP_Y           = 0.07

MIN_CHECKINS     = 20

# load data -----------------------------------------------
data    = pd.read_csv('../input/train.csv')
data.set_index('row_id')

print('Window: (%.2f, %.2f) width: %.2f' % (WINDOW_X, WINDOW_Y, WINDOW_LEN))
data    = data[(data['x'] >= WINDOW_X) & (data['x'] < WINDOW_X+WINDOW_LEN) &
               (data['y'] >= WINDOW_Y) & (data['y'] < WINDOW_Y+WINDOW_LEN)]

N       = data.shape[0]
N_cv    = int(N * CROSS_VALIDATION)
N_train = N - N_cv
    
# split train and test
train = data[:N_train].copy()
cv    = data[N_train:N].copy()

del data

print('Train shape: %s' % str(train.shape))
print('Cross validation: %s' % str(cv.shape))

# data clean up -------------------------------------------
def calculate_time(data):
    print('Calculate hour, weekday, month and year')
    data['hour']    = 1 + (data['time']//60)         % 24
    data['weekday'] = 1 + (data['time']//(60*24))    % 7
    data['month']   = 1 + (data['time']//(60*24*30)) % 12
    data['year']    = 1 + (data['time']//(60*24*365)) 
    return data
   
train = calculate_time(train)
cv    = calculate_time(cv)
    
print('Train shape: %s' % str(train.shape))
print('Cross validation: %s' % str(cv.shape))

places = train['place_id'].unique()
rows_to_drop = []
for place in places:
    rows = train[train['place_id'] == place]['row_id']
    if rows.count() < MIN_CHECKINS:
        rows_to_drop += list(rows.values)

print('Drop %d rows where places have less than %d check-ins' % (len(rows_to_drop), MIN_CHECKINS))
train.drop(rows_to_drop, inplace=True)
print('Train shape: %s' % str(train.shape))

# scoring function by ZFTurbo -----------------------------
def apk(actual, predicted, k=3):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

# do modeling ---------------------------------------------
def model(train, test, classifier, columns):
    score = 0
    score_num = 0
    
    y = WINDOW_Y + STEP_Y
    while y < WINDOW_Y + WINDOW_LEN:
        x = WINDOW_X + STEP_X
        while x < WINDOW_X + WINDOW_LEN:
            #print('X,Y: %.2f,%.2f' % (x,y))
            train_cell = train[(train['x'] >= x-STEP_X) & (train['x'] < x+STEP_X) &
                               (train['y'] >= y-STEP_Y) & (train['y'] < y+STEP_Y)]
                           
            if 'x_y' in columns: train_cell['x_y'] = train_cell.apply(lambda row: row['x'] / (row['y']+0.0001), axis=1)
            if 'y_x' in columns: train_cell['y_x'] = train_cell.apply(lambda row: row['y'] / (row['x']+0.0001), axis=1)

            # classification
            X = train_cell[columns];
            Y = train_cell[['place_id']].values.ravel();
            
            clf = classifier(X, Y)

            # prediction            
            test_cell = test[(test['x'] >= x-STEP_X) & (test['x'] < x+STEP_X) &
                             (test['y'] >= y-STEP_Y) & (test['y'] < y+STEP_Y)]

            if 'x_y' in columns: test_cell['x_y'] = test_cell.apply(lambda row: row['x'] / (row['y']+0.0001), axis=1)
            if 'y_x' in columns: test_cell['y_x'] = test_cell.apply(lambda row: row['y'] / (row['x']+0.0001), axis=1)
            
            X_cv = test_cell[columns];
            Y_cv = test_cell[['place_id']].values.ravel();
            
            # evaluation
            preds = clf.predict_proba(X_cv)
            
            for idx, row in enumerate(preds):
                d = dict(zip(clf.classes_, row))
                top3 = nlargest(3, sorted(d.items()), key=itemgetter(1))
               
                score     += apk([Y_cv[idx]], [top3[0][0], top3[1][0], top3[2][0]], 3)
                score_num += 1
            
            x += STEP_X*2
            
        #print('   Y: %.2f score: %.4f' % (y, score / score_num))
        y += STEP_Y*2

    return score / score_num

def random_forest(X, Y):
    clf = RandomForestClassifier(n_estimators=50, n_jobs = -1)
    clf.fit(X, Y)
    return clf
    
def xgb_classifier(X, Y):
    clf = xgb.XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.2, subsample=0.9, colsample_bytree=0.85, objective='multi:softprob', silent=1)
    clf.fit(X, Y)
    return clf

print('Modeling...')
print('Random Forest: score = %.4f'      % model(train, cv, random_forest,  ['x','y','accuracy','hour', 'weekday', 'month', 'year']))
print('XGBoost classifier: score = %.4f' % model(train, cv, xgb_classifier, ['x','y','accuracy','hour', 'weekday', 'month', 'year']))
#print('XGBoost classifier: score = %.4f' % model(train, cv, xgb_classifier, ['x','y','x_y','y_x','accuracy','hour', 'weekday', 'month', 'year']))
