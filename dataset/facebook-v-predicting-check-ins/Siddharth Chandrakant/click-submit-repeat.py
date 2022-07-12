'''
  Based on: Various Existing scripts. 
  Big thanks to Sandro, David, Alex et al for laying the foundations.
'''

# coding: utf-8
__authors__ = 'Kaggle Overfitting Community'

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

#print('	Reading train.csv')
df_train = pd.read_csv('../input/train.csv',
                        usecols=['row_id','x','y','time','place_id','accuracy'], 
                        index_col = 0)
#print('	Reading test.csv')
df_test = pd.read_csv('../input/test.csv',
                        usecols=['row_id','x','y','time','accuracy'],
                        index_col = 0)

lowest_test_timestamp=df_test.time.min()
df_train['time_delta']=(df_train.time-lowest_test_timestamp)/(60*24*30)

#print('Feature Augmentation')
minute = df_train.time%60
df_train['hour'] = df_train['time']//60
df_train.drop(['time'], axis=1, inplace=True)
df_train['weekday'] = df_train['hour']//24
df_train['month'] = df_train['weekday']//30
df_train['year'] = (df_train['weekday']//365+1)*10.0
df_train['hour'] = ((df_train['hour']%24+1)+minute/60.0)*4.0

pd.options.mode.chained_assignment = None
add_data = df_train[df_train.hour<10]# add data for periodic time that hit the boundary
add_data.hour = add_data.hour+96

add_data2 = df_train[df_train.hour>90]
add_data2.hour = add_data2.hour-96

df_train = df_train.append(add_data)
df_train = df_train.append(add_data2)
del add_data,add_data2

df_train['weekday'] = (df_train['weekday']%7+1)*3.1
df_train['month'] = (df_train['month']%12+1)*2.1
df_train['accuracy'] = np.log10(df_train['accuracy'])*10.0

df_test['time_delta']=(df_test.time-lowest_test_timestamp)/(60*24*30)
minute = df_test['time']%60
df_test['hour'] = df_test['time']//60
df_test.drop(['time'], axis=1, inplace=True)
df_test['weekday'] = df_test['hour']//24
df_test['month'] = df_test['weekday']//30
df_test['year'] = (df_test['weekday']//365+1)*10.0
df_test['hour'] = ((df_test['hour']%24+1)+minute/60.0)*4.0
df_test['weekday'] = (df_test['weekday']%7+1)*3.1
df_test['month'] = (df_test['month']%12+1)*2.1
df_test['accuracy'] = np.log10(df_test['accuracy'])*10.0

print('Generating wrong models. They are just useful to get this job :) ... done')

def calculate_distance(distances):
    return distances ** -2

def process_one_cell(df_cell_train, df_cell_test):
    
    #Working on df_train
    place_counts = df_cell_train.place_id.value_counts()
    mask = (place_counts[df_cell_train.place_id.values] >= 8).values
    df_cell_train = df_cell_train.loc[mask]
    
    #Working on df_test
    row_ids = df_cell_test.index
    
    #Feature engineering on x and y
    df_cell_train.loc[:,'x'] *= 490.0
    df_cell_train.loc[:,'y'] *= 980.0
    df_cell_test.loc[:,'x'] *= 490.0
    df_cell_test.loc[:,'y'] *= 980.0
    
    #Preparing data
    le = LabelEncoder()
    y = le.fit_transform(df_cell_train.place_id.values)
    X = df_cell_train.drop(['place_id'], axis=1).values
    
    #Applying the classifier
    clf = KNeighborsClassifier(n_neighbors=np.floor(np.sqrt(y.size)/8).astype(int), 
                            weights=calculate_distance,metric='manhattan',n_jobs=2)
    clf.fit(X, y)
    y_pred = clf.predict_proba(df_cell_test.values)
    pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:,::-1][:,:3]) 
    
    return pred_labels, row_ids
   
def process_grid(df_train, df_test):
    """
    Iterates over all grid cells, aggregates the results
    """
    size = 10.0
    x_step = 0.5
    y_step = 0.25
    
    x_border_augment = 0.03  
    y_border_augment = 0.015
    
    preds = np.zeros((df_test.shape[0], 3), dtype=int)

    for i in range((int)(size/x_step)):
        
        x_min = x_step * i
        x_max = x_step * (i+1)
        x_min = round(x_min, 4)
        x_max = round(x_max, 4) 
        if x_max == size:
            x_max = x_max + 0.001
            
        df_col_train = df_train[(df_train['x'] >= x_min-x_border_augment) & (df_train['x'] < x_max+x_border_augment)]
        df_col_test = df_test[(df_test['x'] >= x_min) & (df_test['x'] < x_max)]

        for j in range((int)(size/y_step)):
            y_min = y_step * j
            y_max = y_step * (j+1)
            y_min = round(y_min, 4)
            y_max = round(y_max, 4)   
            if y_max == size:
                y_max = y_max + 0.001
                
            df_cell_train = df_col_train[(df_col_train['y'] >= y_min-y_border_augment) & (df_col_train['y'] < y_max+y_border_augment)]
            df_cell_test = df_col_test[(df_col_test['y'] >= y_min) & (df_col_test['y'] < y_max)]
            
            #Applying classifier to one grid cell
            pred_labels, row_ids = process_one_cell(df_cell_train, df_cell_test)

            #Updating predictions
            preds[row_ids] = pred_labels
    
    return preds

def generate_sub(preds):    
    print('Writing submission file')
    #Auxiliary dataframe with the 3 best predictions for each sample
    df_aux = pd.DataFrame(preds, dtype=str, columns=['l1', 'l2', 'l3'])
    #Concatenating the 3 predictions for each sample
    ds_sub = df_aux.l1.str.cat([df_aux.l2, df_aux.l3], sep=' ')
    
    #Writting to csv
    ds_sub.name = 'place_id'
    ds_sub.to_csv('submission_v01.csv', index=True, header=True, index_label='row_id')

preds=process_grid(df_train, df_test)

del df_train, df_test

generate_sub(preds)