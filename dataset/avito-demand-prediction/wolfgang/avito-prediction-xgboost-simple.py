import numpy as np
import pandas as pd
import math
import time
import os.path
import gc
import random

import xgboost as xgb


def print_duration (start_time, msg):
    print("[%d] %s" % (int(time.time() - start_time), msg))
    start_time = time.time()
    return start_time
    
### feature engineering section
def n_hash(s):
    random.seed(hash(s))
    return random.random()
    
def hash_column (row, col):
    if col in row:
        return n_hash(row[col])
    return n_hash('none')

def main():
    start_time = time.time()
    # create a xgboost model
    
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)
            
    # fit the model
    chunksize = 100000
    train = pd.read_csv('../input/train.csv', nrows=chunksize, skiprows=range(1,10))
    train['user_hash'] = train.apply (lambda row: hash_column (row, 'user_id'),axis=1)
    train['region_hash'] = train.apply (lambda row: hash_column (row, 'region'),axis=1)
    
    print(train.groupby(['region_hash']).region_hash.count())
    start_time = print_duration (start_time, "Finished reading")   
    
    
    '''
    # start trining
    train_X = train.as_matrix(columns=['user_id'])
    model.fit(train_X, train['deal_probability'])

    # read test data set
    test = pd.read_csv('../input/test.csv')
    test_X = test.as_matrix(columns=['user_id'])
    start_time = print_duration (start_time, "Finished training, start prediction")   
    # predict the propabilities for binary classes    
    pred = model.predict(test_X)
    
    start_time = print_duration (start_time, "Finished prediction, start store results")    
    submission = pd.read_csv("../input/sample_submission.csv")
    submission['deal_probability'] = pred
    submission.to_csv("submission.csv", index=False)
    start_time = print_duration(start_time, "Finished to store result")
    '''
    
if __name__ == '__main__':
    main()
    
    