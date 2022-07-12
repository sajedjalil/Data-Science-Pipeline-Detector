import sys, os, math
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numbers import Number
from ml_metrics import mapk

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import sklearn.cross_validation as cval

##########################################################
# Input varialbles
file_name = '../input/train.csv'
best_n = 3
n_estimators = 10
cv_num = 10
feat = ['x', 'y', 'accuracy', 'minute', 'hour', 'weekday', 'day', 'month', 'year']

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
def train_test_rough_split(train, label, train_ratio):
	percentile = train[label].quantile(train_ratio)
	test = train[(train[label] >= percentile)]
	train = train[(train[label] < percentile)]
	return train,test
	
# input : {df} in Pandas frame, label in Pandas class label, {train_ratio} in float less than 1
# output: training and testing dataframe
# Split a dataframe based on values of a column
def train_test_accurate_split(train, train_ratio):
	train1 = pd.DataFrame()
	test = pd.DataFrame() 
	unique_place = train['place_id'].unique()
	for element in unique_place:
		temp = train[train['place_id'] == element]
		percentile = temp['time'].quantile(.8)
		test_place = temp[(temp['time'] >= percentile)]
		test = test.append(test_place)
		train_place = temp[(temp['time'] < percentile)]
		train1 = train1.append(train_place)
	return train1, test
	
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
    train['weekday'] = train['hour']//24
    train['month'] = train['weekday']//30
    train['year'] = (train['weekday']//365+1)
    train['hour'] = (train['hour']%24+1)
    train['day'] = (train['weekday']%30+1)
    train['weekday'] = (train['weekday']%7+1)
    train['month'] = (train['month']%12+1)
    
    x_min = 1.5
    x_max = 8.5
    y_min = 1.5
    y_max = 8.5
    
    train = train[(train['x'] >= x_min) & (train['x'] < x_max) &
                        (train['y'] >= y_min) & (train['y'] < y_max)]
    print (train.info())
    print ("%d rows X %d columns" % (train.shape[0], train.shape[1]))
    
    ##############################################################################
    # Split the dataset in test and train
    #train, test = train_test_accurate_split(train, 0.8)
    train, test = cval.train_test_split(train, test_size=0.2, random_state=0)
    test['row_id'] = list(range(1,len(test)+1))
    
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
    importances_total = np.zeros(len(feat))
    
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
        
            trainX = train_grid[feat].values
            trainY = train_grid[['place_id']].values.flatten()
            testX = test_grid[feat].values
            testY = test_grid[['place_id']].values.flatten()
            
            ###############################################################################
            # Training
            print ("Training...")
            start = time.clock()
        
            bclf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, verbose=0)
            #from sklearn.neighbors import KNeighborsClassifier
            #bclf = KNeighborsClassifier()
            #from sklearn.ensemble import AdaBoostClassifier
            #bclf = AdaBoostClassifier(base_estimator=clf, n_estimators=clf.n_estimators)
            #from sklearn.calibration import CalibratedClassifierCV
            #bclf = CalibratedClassifierCV(clf, method='isotonic', cv=cv_num)
            #from sklearn.linear_model import LogisticRegression
            #bclf = LogisticRegression(multi_class='multinomial', solver='newton-cg', n_jobs=-1, verbose=1);
            bclf.fit(trainX,trainY)
        
            training_time = time.clock() - start
        
            ##############################################################################
            # Test classifiers
            print ("Testing...")
            importances = bclf.feature_importances_
            importances_total = importances_total + importances
            
            
            i += 1
            print ("Iter:%d/%d GridX:%.2f-%.2f GridY:%.2f-%.2f" % (i, total_iter, x_min, x_max, y_min, y_max))
            

    test = test.sort_values('row_id')
    
    importances_total = (importances_total/total_iter)
    print (importances_total)
    
    plt.title('Feature Importances')
    plt.bar(range(len(feat)), importances_total, align='center')
    plt.xticks(range(len(feat)), feat, rotation=90)
    plt.xlim([-1, len(feat)])
    plt.tight_layout()
    plt.savefig('feature.png')

if __name__ == "__main__":
    main()