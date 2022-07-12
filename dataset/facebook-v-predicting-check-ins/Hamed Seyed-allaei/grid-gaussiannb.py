# adapted from https://www.kaggle.com/drarfc/facebook-v-predicting-check-ins/script-competition-facebook-v/run/271598

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB


df_train = pd.read_csv('../input/train.csv',
                        usecols=['row_id','x','y','time','place_id','accuracy'], 
                        index_col = 0)
df_test = pd.read_csv('../input/test.csv',
                        usecols=['row_id','x','y','time','accuracy'],
                        index_col = 0)





t_train = np.pi*df_train.time/720
t_test =  np.pi*df_test.time/720

df_train['h_sin'] = np.sin(t_train)
df_train['h_cos'] = np.cos(t_train)
df_test['h_sin'] = np.sin(t_test)
df_test['h_cos'] = np.cos(t_test)

t_train /= 7
df_train['w_sin'] = np.sin(t_train)
df_train['w_cos'] = np.cos(t_train)

t_test /= 7
df_test['w_sin'] = np.sin(t_test)
df_test['w_cos'] = np.cos(t_test)

df_train['accuracy'] = np.log10(df_train['accuracy'])
df_test['accuracy'] = np.log10(df_test['accuracy'])

del t_train, t_test

print('Generating wrong models. They are just useful to get this job :) ... done')
pd.options.mode.chained_assignment = None

def process_one_cell(df_cell_train, df_cell_test):
    
    #Working on df_train
    place_counts = df_cell_train.place_id.value_counts()
    mask = (place_counts[df_cell_train.place_id.values] >= 20).values
    df_cell_train = df_cell_train.loc[mask]
    
    #Working on df_test
    row_ids = df_cell_test.index
    
    #Preparing data
    le = LabelEncoder()
    y = le.fit_transform(df_cell_train.place_id.values)
    X = df_cell_train.drop(['place_id', 'accuracy', 'time'], axis=1).values
    X_test = df_cell_test.drop(['accuracy', 'time'], axis=1).values
                                   

    #Applying the classifier
    clf = GaussianNB()
    clf.fit(X, y, df_cell_train.time)
    y_pred = clf.predict_proba(X_test)
    pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:,::-1][:,:3]) 
    
    return pred_labels, row_ids


def process_grid(df_train, df_test):
    """
    Iterates over all grid cells, aggregates the results
    """
    size = 10.0
    x_step = 0.25
    y_step = 0.02
    
    x_border_augment = x_step / 8
    y_border_augment = y_step / 8
    
    preds = np.zeros((df_test.shape[0], 3), dtype=int)

    for i in range((int)(size/x_step)):
        print(i)
        
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
    #Auxiliary dataframe with the 3 best predictions for each sample
    df_aux = pd.DataFrame(preds, dtype=str, columns=['l1', 'l2', 'l3'])

    print('Writing submission file')
    #Concatenating the 3 predictions for each sample
    ds_sub = df_aux.l1.str.cat([df_aux.l2, df_aux.l3], sep=' ')
    
    #Writting to csv
    ds_sub.name = 'place_id'
    ds_sub.to_csv('grid_gnb.csv', index=True, header=True, index_label='row_id')


preds=process_grid(df_train, df_test)

generate_sub(preds)
