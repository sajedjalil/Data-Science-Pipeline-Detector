# coding: utf-8

"""
Ultra fast parallel, memory efficient implementation of grid KNN model
Takes just 7 minutes for 20x40 grid on 4 core i5-6500, 16GB machine

Based on work of Sandro Vega Pons:
https://www.kaggle.com/svpons/facebook-v-predicting-check-ins/grid-plus-classifier

This is for validation, the script does no create submission
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from os.path import join, exists
import time
import multiprocessing as mp
import gc
import ctypes
from contextlib import closing
from itertools import product


input_dir = "../input"
train_file = "train.pkl"
test_file = "test.pkl"

columns_to_drop = ['place_id', 'time', 'x', 'y']
box_sz = 10

# Filter out id's with less than 5 counts
th = 5

# Obtain top 10 predictions. Can be useful for ensembling
top = 10


def apk(l, v):
    """
    Simplified implementation of Average Mean Precision.
    Operates on a single record.
    Input:
    ------
    l:  list of predictions
    v:  correct value
    """
    return next( (1.0/(i+1) for i,x in enumerate(l[:3]) if x == v), 0)


def create_validation(df):
    """
    Split data to train and validation on time
    0.77/0.23 is the competition train/test proportion
    """
    print('Creating validation set ... ', end="")
    br = int(len(df) * 0.77)
    t = df.time.sort_values(inplace=False)
    br_t = t.iloc[br]
    df_train = df[df.time < br_t].copy()
    df_test = df[df.time >= br_t].copy()   
    print('done.')
    len_train = len(df_train)
    len_test = len(df_test)
    print("Train length %i (%0.2f), test length %i (%0.2f)" % \
        (len_train, len_train/(len_train + len_test),
         len_test, len_test/(len_train + len_test)))
    return df_train, df_test
    

def add_time_features(df):
    """
    Time related features (assuming the time = minutes)
    initial_date = '2014-01-01 01:01', arbitrary decision
    """
    initial_date = pd.Timestamp('2014-01-01 01:01')
    d_times = pd.DatetimeIndex(initial_date + pd.to_timedelta(df.time, unit='m'))
    df['hour'] = d_times.hour + d_times.minute / 60
    df['weekday'] = d_times.weekday
    df['month'] = d_times.month
    df['year'] = d_times.year
    return  df


def append_periodic_time(df):
    add_data = df[df["hour"] < 2.5].copy()
    add_data["hour"] += 24
    add_data2 = df[df["hour"] > 22.5].copy()
    add_data2["hour"] -= 24
    df = df.append(add_data).append(add_data2)   
    return df
    

def add_weights(df):
    df["year"] *= 10
    df["hour"] *= 4
    df["weekday"] *= 3.12
    df["month"] *= 2.12
    df["accuracy"] = np.log10(df["accuracy"]) * 10
    df["x_"] = df["x"] * 465.0 
    df["y_"] = df["y"] * 975.0    
    return df


def calculate_distance(distances):
    return distances ** -2
    
    
def prepare_data(use_disk=True):
    """
    Add features, split to train/validation and dump to disk for future use
    Input:
    ------
    set use_disk=False on a read-only file system
    
    Return:
    -------
    df_train, df_test:  pandas data frames
    """
    if use_disk & exists(join(input_dir, train_file)) & exists(join(input_dir, test_file)):
        df_train = pd.read_pickle(join(input_dir, train_file))
        df_test = pd.read_pickle(join(input_dir, test_file)) 
        return df_train, df_test
        
    print('Loading data ...')
    df = pd.read_csv(join(input_dir, 'train.csv'), dtype={'x':np.float32, 
                                                          'y':np.float32, 
                                                          'accuracy':np.int16,
                                                          'time':np.int,
                                                          'place_id':np.int64}, 
                                                          index_col = 0)
    
    print('Creating datetime features ...')
    df = add_time_features(df)

    df_train, df_test = create_validation(df)
    
    df_train = append_periodic_time(df_train)

    df_train = add_weights(df_train)
    df_test = add_weights(df_test)
    
    if use_disk:
        df_train.to_pickle(join(input_dir, train_file))
        df_test.to_pickle(join(input_dir, test_file))
    
    return df_train, df_test


def get_cell(train, test, x, y, g):
    """
    Slice train/test grids

    Input:
    ------
    train, test:    numpy ndarray
    x, y:           coordinates of a cell
    g:              grid data structure

    Return:
    -------
    cell_train, cell_test:
                    numpy ndarray
    """

    x_low = x * g.x_sz
    x_high = (x + 1) * g.x_sz if x < g.n_cell_x-1 else (x + 2) * g.x_sz 
    y_low = y * g.y_sz
    y_high = (y + 1) * g.y_sz if y < g.n_cell_y-1 else (y + 2) * g.y_sz

    cell_train = train[(train[:, g.pos["xtrain"]] >= x_low-g.x_pad) &
                       (train[:, g.pos["xtrain"]] < x_high+g.x_pad) &
                       (train[:, g.pos["ytrain"]] >= y_low-g.y_pad) & 
                       (train[:, g.pos["ytrain"]] < y_high+g.y_pad)]

    # Filter out infrequent id's from a train cell. Tricky without Pandas
    le = LabelEncoder()
    y = le.fit_transform(cell_train[:, g.pos["idtrain"]])
    d = np.bincount(y)
    goodids = le.inverse_transform(np.where(d > g.th)[0])
    cell_train = cell_train[np.in1d(cell_train[:, g.pos["idtrain"]], goodids), :]
    
    cell_test = test[(test[:, g.pos["xtest"]] >= x_low) &
                     (test[:, g.pos["xtest"]] < x_high) &
                     (test[:, g.pos["ytest"]] >= y_low) & 
                     (test[:, g.pos["ytest"]] < y_high)]
    return cell_train, cell_test  
    
                               
def process_one_cell(x, y, g):
    """
    Input:
    ------
    x, y:  coordinates of a cell
    g:     grid data structure

    Return:
    ------    
    labs:  Data Frame with g.top labels for every check-in in a given cell
    probs: Data Frame with of probabilities of top labels
    """ 
    # Access shared arrays from multithreading environment
    train = np.frombuffer(shared_train).reshape(train_x, train_y)
    test = np.frombuffer(shared_test).reshape(test_x, test_y)
    

    cell_train, cell_test = get_cell(train, test, x, y, g)
    if (len(cell_train) == 0) | (len(cell_test) == 0):
        return None, None
    row_ids = cell_test[:, g.pos["row_id"]].astype(int)

    le = LabelEncoder()
    y = le.fit_transform(cell_train[:, g.pos["idtrain"]])
    X = cell_train[:, g.pos["colstrain"]]

    clf = KNeighborsClassifier(n_neighbors=np.floor((np.sqrt(y.size)/5.3)).astype(int),
                               weights=calculate_distance, metric='manhattan', n_jobs=-1)
    clf.fit(X, y)
    
    X_test = cell_test[:, g.pos["colstest"]]
    y_prob = clf.predict_proba(X_test)

    pred_y = np.argsort(y_prob, axis=1)[:,::-1][:,:g.top]
    pred_labels = le.inverse_transform(pred_y).astype(np.int64)
    
    df = pd.DataFrame(cell_test[:, g.pos["idtest"]], index=row_ids, columns=["place_id"], dtype="int64")
    labs = pd.DataFrame(pred_labels, index=row_ids)
    labs = pd.concat([df, labs], axis=1)
    
    probs = pd.DataFrame(y_prob[np.arange(len(y_prob)).reshape(-1,1), pred_y], index=row_ids)
    probs = pd.concat([df, probs], axis=1)
    
    return labs, probs


class grid(object):
    """
    Grid structure for storing data persistent for a grid
    """
    def __init__(self, n_cell_x, n_cell_y, x_p, y_p, th, top, pos):
        self.n_cell_x = n_cell_x
        self.n_cell_y = n_cell_y
        self.x_sz = box_sz / n_cell_x
        self.y_sz = box_sz / n_cell_y
        self.x_pad = self.x_sz * x_p
        self.y_pad = self.y_sz * y_p
        self.th = th
        self.top = top
        self.pos = pos


def iterate_grid(g, verbose=False):
    """
    Iterator submitted to the process pool in chunks. 
    """
    iteration = 0
    range_y = range(g.n_cell_y)
    range_x = range(g.n_cell_x)
    range_y = range(g.n_cell_y//2, g.n_cell_y//2+1)
    range_x = range(g.n_cell_x//2, g.n_cell_x//2+10)

    for y in range_y:
        for x in range_x:
            iteration += 1
            yield (g, iteration, len(range_x)*len(range_y), verbose, x, y)


def f(z):
    """
    Function to process one cell in one of the workers of the pool.
    """
    g, i, l, verbose, x, y = z
    labs, probs = process_one_cell(x, y, g)
    if ((i+1) % 10 == 0) & verbose:
        print("{}/{} done".format(i+1, l))
    return (labs, probs)
    

def init(shared_train_, shared_test_, train_x_, train_y_, test_x_, test_y_):
    """
    Each worker process will call this initializer when it starts.
    Inherits shared data from parent process
    """
    global shared_train
    global shared_test
    global train_x
    global train_y
    global test_x
    global test_y
    shared_train = shared_train_ # must be inhereted, not passed as an argument
    shared_test = shared_test_
    train_x = train_x_
    train_y = train_y_
    test_x = test_x_
    test_y = test_y_
    

def process_one_grid(g, verbose=False):
    """
    Executes pool of workers on a grid
    Return:
    -------
    df_val:  Data Frame with g.top labels for every check-in in a test set
    df_prob: Data Frame with of probabilities of top labels
    (similar to process_one_cell(...))
    """
    it = iterate_grid(g, verbose)
    with closing(mp.Pool(initializer=init, initargs=(shared_train, shared_test,
                                                     train_x, train_y,
                                                     test_x, test_y))) as p:
        result = p.map(f, it)
    
    labs, probs = [z for z in zip(*result)]
    df_val = pd.concat(labs)
    df_prob = pd.concat(probs)
    df_val = df_val.sort_index()
    df_prob = df_prob.sort_index()
    
    return df_val, df_prob
    

if __name__ == '__main__':
    mp.freeze_support()
    
    # Organize shared data
    global shared_train
    global shared_test
    global train_x
    global train_y
    global test_x
    global test_y   

    df_train, df_test = prepare_data(use_disk=False)    # for read-only file system
    df_test = df_test.reset_index()

    """
    It appears problematic to share and access pandas data frame from Pool 
    threads without copying it. We use numpy arrays instead, and therefore 
    need to preserve column information in pos dictionary to pass it 
    to pool workers as an argument
    """
    cols = df_train.columns.drop(columns_to_drop)    
    pos = {}
    pos["xtrain"] = df_train.columns.get_loc('x')
    pos["ytrain"] = df_train.columns.get_loc('y')
    pos["xtest"] = df_test.columns.get_loc('x')
    pos["ytest"] = df_test.columns.get_loc('y')
    pos["colstrain"] = [df_train.columns.get_loc(c) for c in cols]
    pos["colstest"]  = [df_test.columns.get_loc(c) for c in cols]
    pos["idtrain"] = df_train.columns.get_loc('place_id')
    pos["idtest"] = df_test.columns.get_loc('place_id')
    pos["row_id"] = df_test.columns.get_loc('row_id')    


    # create shared array
    shared_train = mp.RawArray(ctypes.c_double, int(df_train.size))
    train = np.frombuffer(shared_train)
    train[:] = df_train.values.flatten()
    train_x = df_train.shape[0]
    train_y = df_train.shape[1]
    
    shared_test = mp.RawArray(ctypes.c_double, int(df_test.size))
    test = np.frombuffer(shared_test)
    test[:] = df_test.values.flatten()
    test_x = df_test.shape[0]
    test_y = df_test.shape[1]
    
    del df_train, df_test
    gc.collect()

    # Fast implementation allows testing lot of grids
    grids = list(product(range(20, 101, 10), range(20, 101, 10)))

    # Track results in rec
    rec = pd.DataFrame(columns=["grid", "th", "time", "MAP"])

    for i,gr in enumerate(grids):
        print("Processing grid {}. ".format(gr), end="")
        t0 = time.time()
        
        # Fill in grid variables and pass it to pool workers as an argument
        g = grid(*gr, 0.054, 0.06, th, top, pos)
        
        df_val, df_prob = process_one_grid(g, verbose=False)

        # Calculate MAP
        vv = df_val.loc[:, "place_id"].values
        ll = df_val.loc[:, [0, 1, 2]].values
        AP = np.array([apk(l, v) for v,l in zip(vv, ll)])
        MAP = AP.mean()
        
        t1 = time.time()

        print("{} minutes elapsed, MAP={}".format(int(t1 - t0) // 60, MAP))
        rec.loc[i, :] = [gr, th, int(t1 - t0), MAP]
        rec.to_csv("records.csv", index=False)
        
        
        """
        Clear unnecessary data. This script did not use neither df_prob nor 
        extra top predictions, but this can be used in other applications
        """
        del df_val, df_prob
        gc.collect()