import sys, os, math
import time
import datetime
import numpy as np
import pandas as pd
from numbers import Number
from ml_metrics import mapk

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import sklearn.cross_validation as cval

##########################################################
# Input varialbles
file_name = '../input/train.csv'
n_estimators = 50
cv_num = 10
n_best = 5

##########################################################
# Addiotional functions
# input : {target} as Pandas Series (e.g. pd_dataframe[data_label]), {top_index} as top number of types of data occured in INT 
# output: top indicies of elements in Pandas Series
def findTopOccurrences(target, top_index):
    df_targetcounts = target.value_counts()
    df_toptargets = df_targetcounts.iloc[0:top_index]
    return list(df_toptargets.index)

# input : {df} in Pandas frame, label in Pandas class label, {target} in Number or Iterable 
# output: top indicies of elements in Pandas Series
# Find a sub DataFrame(Matrix) from selected target value(s) in the given class labels(columns)
def subDataFrame(df, label, target, flag=0):
    if flag == 0:
        if isinstance(target, Number):
            return df.loc[df[label] == target]
        else:
            return df.loc[df[label].isin(target)]
    else:
        if isinstance(target, Number):
            return df.loc[df[label] != target]
        else:
            return df.loc[~df[label].isin(target)]

# input : {df} in Pandas frame, label in Pandas class label, {train_ratio} in float less than 1
# output: training and testing dataframe
# Split a dataframe based on values of a column
def train_test_split(train, label, train_ratio):
	percentile = train[label].quantile(train_ratio)
	test = train[(train[label] >= percentile)]
	train = train[(train[label] < percentile)]
	return train,test
	
##########################################################
def main():
    # Read from file 
    print ("Reading the file...")
    types = {'row_id': np.int64,
             'x': np.float64,
             'y' : np.float64,
             'accuracy': np.int64,
             'time': np.int64,
             'place_id': np.int64 }

    train = pd.read_csv(os.path.expanduser(file_name), dtype=types)  # load pandas dataframe
    # Select the top occurence in place_id from train
    # temp = findTopOccurrences(train['place_id'], 60)
    # train = subDataFrame(train, 'place_id', temp, flag=0)
    # assume that time is in minutes
    print('Calculate hour, weekday, month and year for data')
    train['minute'] = train['time']%60
    train['hour'] = train['time']//60
    train['day'] = train['hour']//24
    train['month'] = train['day']//30
    train['year'] = train['day']//365+1
    train['hour'] = train['hour']%24+1
    train['day'] = train['day']%7+1
    train['month'] = train['month']%12+1
    
    x_min = 4.6
    x_max = 5.4
    y_min = 4.6
    y_max = 5.4
    
    train = train[(train['x'] >= x_min) & (train['x'] < x_max) &
                        (train['y'] >= y_min) & (train['y'] < y_max)]
    print (train.info())
    print ("%d rows X %d columns" % (train.shape[0], train.shape[1]))
    train = train.reset_index(drop=True)
    
    ##############################################################################
    # Split the dataset in test and train
    # train, test = train_test_split(train, 'time', 0.8)
    train, test = cval.train_test_split(train, test_size=0.2, random_state=0)
    test['row_id'] = list(range(0,len(test)))
    
    ##############################################################################
    # Set up grid
    x_step = 0.2
    y_step = 0.2
    x_ranges = list(zip(np.arange(x_min, x_max, x_step), np.arange(round(x_min+x_step,4), round(x_max+x_step,4), x_step)));
    print (x_ranges)
    y_ranges = list(zip(np.arange(y_min, y_max, y_step), np.arange(round(y_min+y_step,4), round(y_max+y_step,4), y_step)));
    print (y_ranges)
    
    # Initialize values for the loop
    i = 0
    total_iter = len(x_ranges) * len(y_ranges)
    result = pd.DataFrame(columns=['iter', 'x_min', 'x_max', 'y_min', 'y_max', 'training_time', 'testing_time', 'training_size', 'testing_size','map@3'])
    preds = np.zeros((test.shape[0], n_best), dtype=np.int64)
    probs = np.zeros((test.shape[0], n_best), dtype=np.float64)
    
    for x_min, x_max in  x_ranges:
        x_min = round(x_min, 4)
        x_max = round(x_max, 4)
        for y_min, y_max in  y_ranges:
            y_min = round(y_min, 4)
            y_max = round(y_max, 4)
            
            train_grid = train[(train['x'] >= x_min) & (train['x'] < x_max) &
                               (train['y'] >= y_min) & (train['y'] < y_max)]
            original_size = train_grid.shape[0]
            train_grid = train_grid.groupby("place_id").filter(lambda x: len(x) >= cv_num)
            print ("Before: %d rows || After: %d rows" % (original_size, train_grid.shape[0]))
    
            test_grid = test[(test['x'] >= x_min) & (test['x'] < x_max) &
                             (test['y'] >= y_min) & (test['y'] < y_max)]
            row_ids = test_grid.row_id
            
            trainX = train_grid[['x', 'y', 'accuracy', 'minute', 'hour', 'day', 'month', 'year']].values
            trainY = train_grid[['place_id']].values.flatten()
            testX = test_grid[['x', 'y', 'accuracy', 'minute', 'hour', 'day', 'month', 'year']].values
            testY = test_grid[['place_id']].values.flatten()
            
            ###############################################################################
            # Training
            print ("Training...")
            start = time.clock()
        
            clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, verbose=0)
            
            #from sklearn.ensemble import AdaBoostClassifier
            #bclf = AdaBoostClassifier(base_estimator=clf, n_estimators=clf.n_estimators)
            from sklearn.calibration import CalibratedClassifierCV
            bclf = CalibratedClassifierCV(clf, method='isotonic', cv=cv_num)
            
            bclf.fit(trainX,trainY)
        
            training_time = time.clock() - start
        
            ##############################################################################
            # Test classifiers
            print ("Testing...")
            start_testing = time.clock()
            y_pred = bclf.predict_proba(testX)
            pred_labels = bclf.classes_[np.argsort(y_pred, axis=1)[:,::-1][:,:n_best]]
            pred_probas = np.sort(y_pred, axis=1)[:,::-1][:,:n_best]

            preds[row_ids] = pred_labels
            probs[row_ids] = pred_probas
            testing_time = time.clock() - start_testing
            
            trueY = np.array([testY]).T
            score = mapk(trueY, pred_labels, 3)
           
            # trueY, predY = testY, bclf.predict(testX)
            # Classification report
            # print("Accuracy Score: %.3f" % metrics.accuracy_score(trueY, predY))
            # print("F1 Score: %.3f" % metrics.f1_score(trueY, predY, average='micro'))
            #print("Detailed classification report:")
            #print(metrics.classification_report(trueY, predY))
            
            i += 1
            print ("Iter:%d/%d GridX:%.2f-%.2f GridY:%.2f-%.2f" % (i, total_iter, x_min, x_max, y_min, y_max))
            print ("Train - Time:%.3f Size:%d || Test - Time:%.3f Size:%d Map@3:%.5f" % (training_time, train_grid.shape[0], testing_time, test_grid.shape[0], score))
            result.loc[len(result)]=[i,x_min,x_max,y_min,y_max,training_time,testing_time,train_grid.shape[0],test_grid.shape[0], score] 
    
    sub_file = os.path.join('result_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
    result.to_csv(sub_file,index=False)
    del (train)
    del (test)
    #Auxiliary dataframe with the 3 best predictions for each sample
    df_data = pd.DataFrame(preds, dtype=np.int64, columns=['l1', 'l2', 'l3', 'l4', 'l5'])
    df_data.to_csv('preds_rf3.csv', index=True, header=True, index_label='row_id')
    df_data = pd.DataFrame(probs, dtype=float, columns=['p1', 'p2', 'p3', 'p4', 'p5'])    
    df_data.to_csv('proba_rf3.csv', index=True, header=True, index_label='row_id')

if __name__ == "__main__":
    main()