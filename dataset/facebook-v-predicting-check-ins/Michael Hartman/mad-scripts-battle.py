# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:38:36 2016
 
@author: Michael Hartman
"""
 
'''Inspired by several scripts at:
https://www.kaggle.com/c/facebook-v-predicting-check-ins/scripts
Special thanks to Sandro for starting the madness. :-)
https://www.kaggle.com/svpons/facebook-v-predicting-check-ins/grid-plus-classifier
'''
import numpy as np
import pandas as pd
import time
from datetime import timedelta
from sklearn.neighbors import KNeighborsClassifier
 
# Found at: https://www.kaggle.com/rshekhar2/facebook-v-predicting-check-ins/xgboost-cv-example-with-small-bug
def mapkprecision(truthvalues, predictions):
    '''
    This is a faster implementation of MAP@k valid for numpy arrays.
    It is only valid when there is one single truth value. 
 
    m ~ number of observations
    k ~ MAP at k -- in this case k should equal 3
 
    truthvalues.shape = (m,) 
    predictions.shape = (m, k)
    '''
    z = (predictions == truthvalues[:, None]).astype(np.float32)
    weights = 1./(np.arange(predictions.shape[1], dtype=np.float32) + 1.)
    z = z * weights[None, :]
    return np.mean(np.sum(z, axis=1))
 
def load_data(data_name):
    df = pd.read_csv(datapath + data_name, dtype=types, index_col = 0)
    return df
    
def validation_split(df, val_start_day):
    day = df['time']//1440
    df_val = df.loc[(day>=val_start_day)]
    df_train = df.loc[(day<val_start_day)]
    return df_train, df_val
    
# Generate list of cells and get predictions
def process_grid(fw, th):
    pred_list = []
    cells = range(x_cuts * y_cuts)
    for cell in cells:
        x_cut = cell // y_cuts
        y_cut = cell % y_cuts        
        dim_slice = 10 / x_cuts
        x_train_filter, x_test_filter = get_filters(x_cut, dim_slice, x_cuts, 
                                                    x_border_aug, 'x')
        
        dim_slice = 10 / y_cuts
        y_train_filter, y_test_filter = get_filters(y_cut, dim_slice, y_cuts,
                                                    y_border_aug, 'y')
       
        df_cell = df_train[x_train_filter & y_train_filter].copy()
        df_cell['x'] *= fw[0]
        df_cell['y'] *= fw[1]
        df_cell_train = df_cell
        df_cell = df_test[x_test_filter & y_test_filter].copy()
        df_cell['x'] *= fw[0]
        df_cell['y'] *= fw[1]
        df_cell_test = df_cell
        cell = (df_cell_train, df_cell_test)
        pred = process_one_cell(cell, fw, th)
        pred_list.append(pred)
    preds = np.vstack(pred_list)
    return preds
 
def process_one_cell(cell, fw, th):
 
    df_cell_train, df_cell_test = cell
    # Remove infrequent places
    place_counts = df_cell_train.place_id.value_counts()
    mask = (place_counts[df_cell_train.place_id.values] >= th).values
    df_cell_train = df_cell_train.loc[mask].copy()
    
    # Store row_ids for test
    row_ids = df_cell_test.index.values
    
    y = df_cell_train['place_id'].values.astype(np.int64)
    X = df_cell_train.drop(['place_id'], axis=1).values
    X_test = df_cell_test.values
 
    #Applying the classifier
    n_neighbors=np.floor(np.sqrt(y.size)/5.1282).astype(int)
    #print('n_neighbors:', n_neighbors)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, 
                               weights=calculate_distance, 
                               metric='manhattan', n_jobs=1)
    clf.fit(X, y)
    y_pred = clf.predict_proba(X_test)
    v_places = np.vectorize(lambda x: clf.classes_[x])
    pred_labels = v_places(np.argsort(y_pred, axis=1)[:,:-4:-1])
    cell_pred = np.column_stack((row_ids, pred_labels)).astype(np.int64)
    
    return cell_pred
 
def calculate_distance(distances):
    return distances ** -2
    
def get_filters(cut, dim_slice, dim_cuts, dim_border_aug, dim):
    dim_min = cut * dim_slice
    dim_max = (cut + 1) * dim_slice + int(cut + 1 == dim_cuts) * 0.001
    dim_test_filter = (df_test[dim] >= dim_min) & (df_test[dim] < dim_max)
    dim_min -= dim_border_aug
    dim_max += dim_border_aug
    dim_train_filter = df_train[dim].between(dim_min, dim_max)
    return dim_train_filter, dim_test_filter
    
def feature_engineering(df, fw):
    minute = df.time%60
    df['hour'] = df['time']//60
    df['weekday'] = df['hour']//24
    df['month'] = df['weekday']//30
    df['year'] = (df['weekday']//365+1)*fw[5]
    df['hour'] = ((df['hour']%24+1)+minute/60.0)*fw[2]
    df['weekday'] = ((df['weekday']%7)+1)*fw[3]
    df['month'] = (df['month']%12+1)*fw[4]
    df['accuracy'] = np.log10(df['accuracy'])*fw[6]
    return df
    
# Adapted from: https://www.kaggle.com/ma350365879/facebook-v-predicting-check-ins/script-competition-facebook-v
# and from: https://www.kaggle.com/drarfc/facebook-v-predicting-check-ins/script-competition-facebook-v/run/273652/code
def time_engineering(df):
    add_data = df[df.hour<10].copy()
    add_data.hour = add_data.hour+96
    df = df.append(add_data)
    add_data = df[df.hour>90].copy()
    add_data.hour = add_data.hour-96
    df = df.append(add_data)
    del add_data
    return df
            
if __name__ == '__main__':
    """
    """
    # Global variables
    types = {'row_id': np.dtype(np.int32),
             'x': np.dtype(float),
             'y' : np.dtype(float),
             'accuracy': np.dtype(np.int16),
             'place_id': np.int64,
             'time': np.dtype(np.int32)}
    datapath = '../input/'
    fw = [490., 980., 4., 3., 2., 10., 10.]
#    fw = [6243., 2504., 12., 21.11, 3.32, 62.43, 51.28] #feature weights (black magic here)
    th = 8 # Threshold of minimum instances of a place to use it in training
#    n_neighbors = 31
    val_start_day = 455 # Day at which to cut validation
    # Set validation to zero to generate predictions
    
    # Defining the size of the grid
    x_cuts = 20 # number of cuts along x 
    y_cuts = 20 # number of cuts along y
    x_border_aug = 0.03 # expansion of x border on train 
    y_border_aug = 0.015 # expansion of y border on train 
    print('Starting...')
    start_time = time.time()
    df_train = load_data('train.csv')
    df_train = feature_engineering(df_train, fw)
    if val_start_day > 0:
        # Create validation data
        df_train, df_test = validation_split(df_train, val_start_day)
        val_label = df_test['place_id']
        df_train.drop(['time'], axis=1, inplace=True)
        df_test.drop(['place_id', 'time'], axis=1, inplace=True)
    else:
        df_train.drop(['time'], axis=1, inplace=True)
        df_test = load_data('test.csv')
        df_test = feature_engineering(df_test, fw)
        df_test.drop(['time'], axis=1, inplace=True)
    elapsed = (time.time() - start_time)
    print('Data prepared in:', timedelta(seconds=elapsed))
    
    print('Processing grid...')
    preds = process_grid(fw, th)
    elapsed = (time.time() - start_time)
    print('Predictions made in:', timedelta(seconds=elapsed))
    
    if val_start_day > 0:
        # Generate the score for the validation
        preds = preds[preds[:, 0] > 0] # only use rows predicted
        labels = val_label.loc[preds[:, 0]].values
        score = mapkprecision(labels, preds[:, 1:])
        print('Final score:', score)
    else:
        cols = ['pred'+str(i) for i in range(preds.shape[1] - 1)]
        preds = pd.DataFrame(preds[:, 1:], dtype=str, 
                                index=preds[:,0], columns=cols)
        preds.index.name = 'row_id'
        cat_cols = [preds[i] for i in cols[1:]]
        preds['place_id'] = preds[cols[0]].str.cat(cat_cols, sep=' ')
        preds['place_id'].to_csv('knn_sub.csv', index=True, header=True)
    elapsed = (time.time() - start_time)
    print('Task completed in:', timedelta(seconds=elapsed))