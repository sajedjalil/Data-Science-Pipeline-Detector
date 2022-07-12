# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

'''Partially based on several scripts:
https://www.kaggle.com/c/facebook-v-predicting-check-ins/scripts)
'''

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
    df['grid_cell_x'] = pos_x
    df['grid_cell_y'] = pos_y
    
    #Feature engineering
    df['accuracy'] = df['accuracy'].apply(np.log10)
    initial_date = np.datetime64('2014-01-01T01:01', dtype='datetime64[m]') 
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') 
                               for mn in df.time.values)

    df['hour'] = d_times.hour + d_times.minute / 60.
    df['weekday'] = d_times.weekday
    
    df = df.drop(['time'], axis=1) 
    return df
    
def process_one_cell(df_train, df_test, gx_id, gy_id, x_border, y_border, th):
    """   
    Classification inside one grid cell.
    """
    #Working on df_train
    #filtering occurance smaller than th
    #consider border of cell
    df_cell_train = df_train.loc[(df_train.grid_cell_x == gx_id) & (df_train.grid_cell_y == gy_id)]
    x_min = df_cell_train.x.min()
    x_max = df_cell_train.x.max()
    y_min = df_cell_train.y.min()
    y_max = df_cell_train.y.max()
    df_cell_train = df_train.loc[(df_train.x >= x_min - x_border) & (df_train.x <= x_max + x_border)
                                  & (df_train.y >= y_min - y_border) & (df_train.y <= y_max + y_border)]
    place_counts = df_cell_train.place_id.value_counts()
    mask = (place_counts[df_cell_train.place_id.values] >= th).values
    df_cell_train = df_cell_train.loc[mask]

    #Working on df_test
    df_cell_test = df_test.loc[(df_test.grid_cell_x == gx_id) & (df_test.grid_cell_y == gy_id)]
    row_ids = df_cell_test.index
    
    #Preparing data
    le = LabelEncoder()
    y_train = le.fit_transform(df_cell_train.place_id.values)
    
    df_cell_train_feats = df_cell_train.drop(['place_id', 'grid_cell_x', 'grid_cell_y'], axis=1)
    feats = df_cell_train_feats.columns.values
    df_cell_test_feats = df_cell_test[feats]

    #Applying the classifier
    clf = GaussianNB()
    clf.fit(df_cell_train_feats, y_train)
    y_pred = clf.predict_proba(df_cell_test_feats)
    pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:,::-1][:,:3]) 
    
    return pred_labels, row_ids
    
def process_grid(df_train, df_test, th, n_cell_x, n_cell_y, x_border, y_border):
    """
    Iterates over all grid cells, aggregates the results and makes the
    submission.
    """ 
    preds = np.zeros((df_test.shape[0], 3), dtype=int)
    
    n_cell_xs = range(n_cell_x)
    n_cell_ys = range(n_cell_y)
    
    for gx_id in n_cell_xs:
        if gx_id % 10 == 0: print('gx_id: %s' %(gx_id))
        for gy_id in n_cell_ys:
            #Applying classifier to one grid cell
            pred_labels, row_ids = process_one_cell(df_train, df_test, gx_id, gy_id,
                                                    x_border, y_border, th)
            #Updating predictions
            preds[row_ids] = pred_labels

    print('Generating submission file ...')
    #Auxiliary dataframe with the 3 best predictions for each sample
    df_aux = pd.DataFrame(preds, dtype=str, columns=['l1', 'l2', 'l3'])  
    
    #Concatenating the 3 predictions for each sample
    ds_sub = df_aux.l1.str.cat([df_aux.l2, df_aux.l3], sep=' ')
    
    #Writting to csv
    ds_sub.name = 'place_id'
    ds_sub.to_csv('sub_nb.csv', index=True, header=True, index_label='row_id') 
    
if __name__ == '__main__':
    print('Loading data ...')
    df_train = pd.read_csv('../input/train.csv',
                           usecols=['row_id','x','y','accuracy','time','place_id'], 
                           index_col = 0)
    df_test = pd.read_csv('../input/test.csv',
                          usecols=['row_id','x','y','accuracy','time'],
                          index_col = 0)
 
    #Defining the size of the grid
    n_cell_x = 20
    n_cell_y = 40
    x_border = 0.03
    y_border = 0.015
    
    print('Preparing train data')
    df_train = prepare_data(df_train, n_cell_x, n_cell_y)
    
    print('Preparing test data')
    df_test = prepare_data(df_test, n_cell_x, n_cell_y)
    
    #Solving classification problems inside each grid cell
    th = 5 #Keeping place_ids with more than th samples.   
    process_grid(df_train, df_test, th, n_cell_x, n_cell_y, x_border, y_border)