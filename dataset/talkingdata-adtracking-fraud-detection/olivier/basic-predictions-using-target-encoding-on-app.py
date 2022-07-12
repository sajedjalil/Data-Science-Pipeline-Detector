"""
This notebook is part of an effort to support students from DiveIntoCode https://diveintocode.jp/ai_curriculum
The goal is to show how to go through a csv file using pandas with chunksize
No rocket science here, just an iteration over chunks and a groupby statement to compute averages
Test file is then read and predictions are made using the app feature and a map statement
"""

import numpy as np 
import pandas as pd 
import gc
import time


def get_types():
    return {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8'
    }


def create_app_averages():
    """ read the train data file and create averages per app and the target prior """
    app_averages = None
    target_sum = 0.0
    nb_samples = 0.0
    start_time=time.time()
    chunksize=20000000
    train_file_path = "../input/train.csv"
    for i_chunk, df in enumerate(pd.read_csv(train_file_path, 
                                             chunksize=chunksize, 
                                             dtype=get_types(), 
                                             usecols=['app', 'is_attributed'])):
        # Make the groupby statement
        # The groupby statement uses sum and count to be able to compute averages over all samples
        the_group = df.groupby("app").agg(['sum', 'count'])
        the_group.columns = the_group.columns.droplevel(0)
        if app_averages is None:
            app_averages = the_group
        else:
            # pandas .add method makes sure apps that are not in both the_group and app_average
            # take value of 0 before the addition takes place
            app_averages = the_group.add(app_averages, fill_value=0.0)
    
        # Udpate target sum and nb samples for prior calculation
        target_sum += df['is_attributed'].sum()
        nb_samples += df.shape[0]
        
        # Free memory by deleting the current DataFrame
        del df, the_group
        gc.collect()
        
        # Display the time we spent so far
        print("%3d chunks of train.csv have been read in %5.1f minute" 
              % (i_chunk + 1, (time.time() - start_time) / 60))
              
        app_averages['average'] = app_averages['sum'] / app_averages['count'] 
        
    return app_averages, target_sum / nb_samples


def create_test_predictions(app_averages, prior):
    """ read the test data file and create predictions using the app_averages """
    start_time=time.time()
    # Create place holder for predictions
    predictions = None
    test_file_path = '../input/test.csv'
    chunksize = 5000000
    # Read the test file by chunks
    for i_chunk, df in enumerate(pd.read_csv(test_file_path, 
                                             chunksize=chunksize, 
                                             dtype=get_types(), 
                                             usecols=['click_id', 'app'])):
        if predictions is None:
            predictions = df[['click_id']] # double square brackets are used to return a DataFrame and not a Series
            predictions['is_attributed'] = df['app'].map(app_averages['average']).astype(np.float32)
        else:
            curr_preds = df[['click_id']] # double square brackets are used to return a DataFrame and not a Series
            curr_preds['is_attributed'] = df['app'].map(app_averages['average']).astype(np.float32)
            # Stack predictions and current predictions
            predictions: pd.DataFrame = pd.concat([predictions, curr_preds], axis=0)
            # free memory
            del curr_preds
            
        # Free memory by deleting the current DataFrame
        del df
        gc.collect()
        
        # Display the time we spent so far
        print("%3d chunks of test.csv have been read in %5.1f minute" 
              % (i_chunk + 1, (time.time() - start_time) / 60))
    
    # For any app in test that is not available in train, give prediction the value of the prior
    return predictions.fillna(prior)


def main():
    app_averages, prior = create_app_averages()
    predictions = create_test_predictions(app_averages, prior)
    predictions.to_csv('app_encoding_submission.csv', index=False, float_format='%.7f')
    
    
if __name__ == '__main__':
    gc.enable()
    main()
