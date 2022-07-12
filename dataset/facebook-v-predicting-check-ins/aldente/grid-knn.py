# coding: utf-8
__author__ = 'Sandro Vega Pons : https://www.kaggle.com/svpons'

'''Partially based on grid_plus_classifier script:
https://www.kaggle.com/svpons/facebook-v-predicting-check-ins/grid-plus-classifier

Adding some changes by David, Alexandru Papiu, aldente
'''

import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier


fw = [500, 1000, 4, 3, 2, 10] #feature weights (black magic here)

def prepare_data(df):
    """
    Feature engineering and computation of the grid.
    """
    
    #Feature engineering
    df.x = df.x.values * fw[0]
    df.y = df.y.values * fw[1]
    initial_date = np.datetime64('2014-01-01T01:01', dtype='datetime64[m]') 
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') 
                               for mn in df.time.values)    
    df['hour'] = (d_times.hour+ d_times.minute/60) * fw[2]
    df['weekday'] = d_times.weekday * fw[3]
    df['month'] = d_times.month * fw[4]
    df['year'] = (d_times.year - 2013) * fw[5]
    
    df = df.drop(['time'], axis=1) 
    return df
    

def process_one_cell(df_train, df_test, x_min, x_max, y_min, y_max, th):
    """   
    Classification inside one grid cell.
    """  
    x_border_augment = 0.025
    y_border_augment = 0.0125

    #Working on df_train
    df_cell_train = df_train[(df_train['x'] >= x_min-x_border_augment) & (df_train['x'] < x_max+x_border_augment) &
                               (df_train['y'] >= y_min-y_border_augment) & (df_train['y'] < y_max+y_border_augment)]
    place_counts = df_cell_train.place_id.value_counts()
    mask = (place_counts[df_cell_train.place_id.values] >= th).values
    
    df_cell_train = df_cell_train.loc[mask]
    
    #Working on df_test
    # to be delete: df_cell_test = df_test.loc[df_test.grid_cell == grid_id]
    df_cell_test = df_test[(df_test['x'] >= x_min) & (df_test['x'] < x_max) &
                               (df_test['y'] >= y_min) & (df_test['y'] < y_max)]
    row_ids = df_cell_test.index

    #Feature engineering on x and y
    df_cell_train.loc[:,'x'] *= fw[0]
    df_cell_train.loc[:,'y'] *= fw[1]
    df_cell_test.loc[:,'x'] *= fw[0]
    df_cell_test.loc[:,'y'] *= fw[1]
    
    #Preparing data
    le = LabelEncoder()
    y = le.fit_transform(df_cell_train.place_id.values)
    X = df_cell_train.drop(['place_id'], axis=1).values.astype(float)
    X_test = df_cell_test.values.astype(float)

    #Applying the classifier
    clf = KNeighborsClassifier(n_neighbors=32, weights='distance', 
                               metric='manhattan', n_jobs=-1)
    clf.fit(X, y)
    y_pred = clf.predict_proba(X_test)
    pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:,::-1][:,:3]) 
    
    return pred_labels, row_ids
   
   
def process_grid(df_train, df_test, th, size, x_step, y_step):
    """
    Iterates over all grid cells, aggregates the results and makes the
    submission.
    """ 
    preds = np.zeros((df_test.shape[0], 3), dtype=int)
    
    for i in range((int)(size/x_step)):
        start_time_row = time.time()
        x_min = x_step * i
        x_max = x_step * (i+1)
        x_min = round(x_min, 4)
        x_max = round(x_max, 4) 
        if x_max == size:
            x_max = x_max + 0.001

        for j in range((int)(size/y_step)):
            y_min = y_step * j
            y_max = y_step * (j+1)
            y_min = round(y_min, 4)
            y_max = round(y_max, 4)   
            if y_max == size:
                y_max = y_max + 0.001

            #Applying classifier to one grid cell
            pred_labels, row_ids = process_one_cell(df_train, df_test, x_min, x_max, y_min, y_max, th)

            #Updating predictions
            preds[row_ids] = pred_labels

        print("Row %d/%d elapsed time: %.2f seconds" % (i+1, (int)(size/x_step),(time.time() - start_time_row)))

    print('Generating submission file ...')
    #Auxiliary dataframe with the 3 best predictions for each sample
    df_aux = pd.DataFrame(preds, dtype=str, columns=['l1', 'l2', 'l3'])  
    
    #Concatenating the 3 predictions for each sample
    ds_sub = df_aux.l1.str.cat([df_aux.l2, df_aux.l3], sep=' ')
    
    #Writting to csv
    ds_sub.name = 'place_id'
    ds_sub.to_csv('sub_knn.csv', index=True, header=True, index_label='row_id')  
      

if __name__ == '__main__':
    """
    """
    print('Loading data ...')
    df_train = pd.read_csv('../input/train.csv',
                           usecols=['row_id','x','y','time','place_id'], 
                           index_col = 0)
    df_test = pd.read_csv('../input/test.csv',
                          usecols=['row_id','x','y','time'],
                          index_col = 0)
 
    #Defining the size of the grid
    size = 10.0
    x_step = 0.5
    y_step = 0.25
    
    print('Preparing train data')
    df_train = prepare_data(df_train)
    
    print('Preparing test data')
    df_test = prepare_data(df_test)
    
    #Solving classification problems inside each grid cell
    th = 5 #Keeping place_ids with more than th samples.   
    process_grid(df_train, df_test, th, size, x_step, y_step)