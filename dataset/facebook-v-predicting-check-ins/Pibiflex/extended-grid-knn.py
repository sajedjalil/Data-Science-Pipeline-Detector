# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# coding: utf-8
__author__="pibiflex"
__previous_author__ = 'Sandro Vega Pons : https://www.kaggle.com/svpons'

'''Partially based on grid_plus_classifier script:
https://www.kaggle.com/svpons/facebook-v-predicting-check-ins/grid-plus-classifier

Feature changed : only time of the day and day are taken, weights changed
Training set extension : for each training (in a cell) the set is extended with the neighbors: in the dx and/or dy border and data are copied over the two time limits 0 and 1440

TODO : change classifier and weights
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier


def prepare_data(df, n_cell_x, n_cell_y):
    """
    Feature engineering and computation of the grid.
    """
    #Creating the grid
    size_x = 10. / n_cell_x
    size_y = 10. / n_cell_y
    eps = 0.00001  
    xs = np.where(df.x.values < eps, 0, df.x.values - eps)
    ys = np.where(df.y.values < eps, 0, df.y.values - eps)
    pos_x = (xs / size_x).astype(np.int)
    pos_y = (ys / size_y).astype(np.int)
    df['grid_cell'] = pos_y * n_cell_x + pos_x
    
    #Feature engineering
    fw = [1,1.6,0.00012,0.0005]
    df.x = df.x.values * fw[0]
    df.y = df.y.values * fw[1]  
    df['hour'] = df.time.values // 1440 * fw[3]
    df.time = df.time.values % 1440 * fw[2]

    return df
   
def extend_time(df):
    '''
    time between 0 and 720 is put from 1440 to 1440+720 and time between 720 and 1440 is put between -720 and 0
    '''
    out=df.copy()
    out.time = ( df.time.values + 3*720 ) % (4*720) - 720
    return out
    
def list_of_grid_neighbors(grid_id, n_cell_x, n_cell_y):
    out=[]
    if grid_id % n_cell_x + 1 < n_cell_x :
        out.append(grid_id + 1)
    if grid_id % n_cell_x > 0 :
        out.append(grid_id - 1)
    if grid_id // n_cell_x > 0:
        out.append(grid_id - n_cell_x)
    if grid_id // n_cell_x + 1 < n_cell_y:
        out.append(grid_id + n_cell_x)
    return out

def process_one_cell(df_train, df_test, grid_id, th, list_neighbors):
    """   
    Classification inside one grid cell.
    """   
    #Working on df_train
    df_cell_train = df_train.loc[df_train.grid_cell == grid_id]
    for g in list_neighbors :
        df_cell_train.add(df_train.loc[df_train.grid_cell == g])
    place_counts = df_cell_train.place_id.value_counts()
    mask = (place_counts[df_cell_train.place_id.values] >= th).values
    df_cell_train = df_cell_train.loc[mask]

    #Working on df_test
    df_cell_test = df_test.loc[df_test.grid_cell == grid_id ]
    row_ids = df_cell_test.index
    
    #Preparing data
    le = LabelEncoder()
    y = le.fit_transform(df_cell_train.place_id.values)
    X = df_cell_train.drop(['place_id', 'grid_cell'], axis=1).values.astype(float)
    X_test = df_cell_test.drop(['grid_cell'], axis = 1).values.astype(float)
    
    
    #Applying the classifier
    clf = KNeighborsClassifier(n_neighbors=40,
                           algorithm='auto',
                           weights='distance',
                           metric='minkowski',
                           p=1,
                           metric_params=None, 
                           leaf_size=40, 
                           n_jobs=1)
    clf.fit(X, y)
    y_pred = clf.predict_proba(X_test)
    pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:,::-1][:,:3]) 
    return pred_labels, row_ids
   
   
def process_grid(df_train, df_test, th, n_cell_x, n_cell_y):
    """
    Iterates over all grid cells, aggregates the results and makes the
    submission.
    """ 
    n_cells = n_cell_x * n_cell_y
    preds = np.zeros((df_test.shape[0], 3), dtype=int)
    
    for g_id in range(n_cells):
        if g_id % 100 == 0 :
            print('iter: %s' %(g_id))
        
        #Applying classifier to one grid cell
        pred_labels, row_ids = process_one_cell(df_train, df_test, g_id, th, list_of_grid_neighbors(g_id, n_cell_x, n_cell_y))

        #Updating predictions
        preds[row_ids] = pred_labels

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
    n_cell_x = 10
    n_cell_y = 20
    
    print('Preparing train data')
    df_train = prepare_data(df_train, n_cell_x, n_cell_y)
    df_train.add(extend_time(df_train))
    
    print('Preparing test data')
    df_test = prepare_data(df_test, n_cell_x, n_cell_y)
    
    #Solving classification problems inside each grid cell
    th = 5 #Keeping place_ids with more than th samples.   
    process_grid(df_train, df_test, th, n_cell_x, n_cell_y)