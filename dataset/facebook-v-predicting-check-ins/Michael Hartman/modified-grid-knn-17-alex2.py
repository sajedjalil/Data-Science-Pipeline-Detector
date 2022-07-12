# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#COMPETITION NAME (the 'Competition'): Facebook V: Predicting Check Ins 
#COMPETITION SPONSOR: Facebook
#COMPETITION WEBSITE: https://www.kaggle.com/c/facebook-v-predicting-check-ins

'''Inspired by several scripts at:
https://www.kaggle.com/c/facebook-v-predicting-check-ins/scripts
Special thanks to
Michael Hartman
https://www.kaggle.com/zeroblue/facebook-v-predicting-check-ins/mad-scripts-battle-z/
Sandro for starting the madness. :-)
https://www.kaggle.com/svpons/facebook-v-predicting-check-ins/grid-plus-classifier
ZFTurbo for the grid concept
https://www.kaggle.com/zfturbo/facebook-v-predicting-check-ins/msb-with-validation

https://www.kaggle.com/drarfc/facebook-v-predicting-check-ins/script-competition-facebook-v-2-1/run/281710/code

Updated version that uses numpy in place of pandas
Uses less memory and may be a bit faster.  

'''

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import time
from datetime import timedelta
import gc
from numba import jit

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
    types = {'row_id': np.dtype(np.int32),
         'x': np.dtype(float),
         'y' : np.dtype(float),
         'accuracy': np.dtype(np.int16),
         'place_id': np.int64,
         'time': np.dtype(np.int32)}
    df = pd.read_csv(data_name, dtype=types, na_filter=False)
    return df

def process_one_cell(cell_train, cell_test, fw, th):
    
    # Remove infrequent places
    cell_train = remove_infrequent_places(cell_train, th)
    
    # Store row_ids for test
    row_ids = cell_test[:, -1].flatten().astype(np.int32)
    cell_test = cell_test[:, :-1]
    
    # Preparing data
    y = cell_train[:, -1].flatten().astype(np.int64)
    X = cell_train[:, :-1]
    
    #Applying the classifier
    cte = 5.8
    n_neighbors = int((y.size ** 0.5) / cte)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors,
                            weights=calculate_distance, p=1, 
                            n_jobs=2, leaf_size=14)
    clf.fit(X, y)
    y_pred = clf.predict_proba(cell_test)
    y_pred_labels = np.argsort(y_pred, axis=1)[:,:-4:-1]
    pred_labels = clf.classes_[y_pred_labels]
    cell_pred = np.column_stack((row_ids, pred_labels)).astype(np.int64) 
    
    return cell_pred
    
def calculate_distance(distances):
    return distances ** -2.22
    
# Precompute the trig values
def time_trig(max_time):
    time_array = np.linspace(0, 2*np.pi, max_time)
    sin_values = np.sin(time_array)
    cos_values = np.cos(time_array)
    return (sin_values, cos_values)

# Generate a dictionary of the time limits so it doesn't have to be 
# recalculated each loop
def create_time_dict(t_cuts, time_mod, time_weight, time_aug):
    
    #print(t_cuts, time_factor, time_aug)
    t_slice = 24 / t_cuts
    time_dict = dict()
    trig_array = time_trig(time_mod)
    for t in range(t_cuts):
        
        t_min = int(t * t_slice * 12)
        t_max = int((t + 1) * t_slice * 12 - 1)
        sin_t_start = trig_array[0][t_min] * time_weight
        sin_t_stop = trig_array[0][t_max] * time_weight
        cos_t_start = trig_array[1][t_min] * time_weight
        cos_t_stop = trig_array[1][t_max] * time_weight
        #print(t, time_weight, (sin_t_start, sin_t_stop, cos_t_start, cos_t_stop))
        sin_t_min = min((sin_t_start, sin_t_stop))
        sin_t_max = max((sin_t_start, sin_t_stop))
        cos_t_min = min((cos_t_start, cos_t_stop))
        cos_t_max = max((cos_t_start, cos_t_stop))
        time_dict[t] = [sin_t_min, sin_t_max, cos_t_min, cos_t_max]

        t_min = int((t * t_slice - time_aug) * 12)%time_mod
        t_max = int(((t + 1) * t_slice + time_aug)* 12 - 1)%time_mod
        sin_t_start = trig_array[0][t_min] * time_weight
        sin_t_stop = trig_array[0][t_max] * time_weight
        cos_t_start = trig_array[1][t_min] * time_weight
        cos_t_stop = trig_array[1][t_max] * time_weight
        sin_t_min = min((sin_t_start, sin_t_stop, sin_t_min))
        sin_t_max = max((sin_t_start, sin_t_stop, sin_t_max))
        cos_t_min = min((cos_t_start, cos_t_stop, cos_t_min))
        cos_t_max = max((cos_t_start, cos_t_stop, cos_t_max))
        time_dict[t] += [sin_t_min, sin_t_max, cos_t_min, cos_t_max]
        
    return time_dict

@jit
def apply_mask(data, feature, mask_min, mask_max):
    mask = (data[:, feature] >= mask_min)
    mask = mask & (data[:, feature] < mask_max)      
    return data[mask]    

def process_grid(train, test, x_cuts, y_cuts, t_cuts,
                 x_border_aug, y_border_aug, time_aug, fw, th):
    preds_list = []
    x_slice = train[:, 0].max() / x_cuts
    y_slice = train[:, 1].max() / y_cuts
    time_mod = 288
    time_weight = fw[2]
    time_dict = create_time_dict(t_cuts, time_mod, time_weight, time_aug)

    for i in range(x_cuts):
        row_start_time = time.time()
        x_min = x_slice * i
        x_max = x_slice * (i+1)
        x_max += int((i+1) == x_cuts) # expand edge at end

        col_test = apply_mask(test, 0, x_min, x_max)
        #print('Test filtered by x')
        x_min -= x_border_aug
        x_max += x_border_aug
        col_train = apply_mask(train, 0, x_min, x_max)
        #print('Train filtered by x')

        for j in range(y_cuts):
            y_min = y_slice * j
            y_max = y_slice * (j+1)
            y_max += int((j+1) == y_cuts) # expand edge at end

            row_test = apply_mask(col_test, 1, y_min, y_max)
            y_min -= y_border_aug
            y_max += y_border_aug
            row_train = apply_mask(col_train, 1, y_min, y_max)

            for t in range(t_cuts):
                #print(df_row_test.shape, df_row_train.shape)
                t_lim = time_dict[t]
                mask = (row_test[:, 2] >= t_lim[0])
                mask = mask & (row_test[:, 2] <= t_lim[1])
                mask = mask & (row_test[:, 3] >= t_lim[2])
                mask = mask & (row_test[:, 3] <= t_lim[3])
                cell_test = row_test[mask]
                mask = (row_train[:, 2] >= t_lim[4])
                mask = mask & (row_train[:, 2] <= t_lim[5])
                mask = mask & (row_train[:, 3] >= t_lim[6])
                mask = mask & (row_train[:, 3] <= t_lim[7])
                cell_train = row_train[mask]
                cell_pred = process_one_cell(cell_train, cell_test, 
                                             fw, th)
                preds_list.append(cell_pred)
        elapsed = (time.time() - row_start_time)
        print('Row', i, 'completed in:', timedelta(seconds=elapsed))
    preds = np.vstack(preds_list)
    return preds

# Thank you Alex!
# From: https://www.kaggle.com/drarfc/facebook-v-predicting-check-ins/fastest-way-to-write-the-csv
def generate_submission(preds):    
    print('Writing submission file')
    print('Pred shape:', preds.shape)
    with open('KNN_submission.csv', "w") as out:
        out.write("row_id,place_id\n")
        rows = ['']*8607230
        for num in range(8607230):
            rows[num]='%d,%d %d %d\n' % (preds[num,0],preds[num,1],preds[num,2],preds[num,3])
        out.writelines(rows)

def validation_split(df, val_start_day):
    day = df['time']//1440
    df_val = df.loc[(day>=val_start_day)].copy()
    df = df.loc[(day<val_start_day)].copy()
    return df, df_val

def remove_infrequent_places(data, th=5):
    places, idx, counts = np.unique(data[:, -1], return_inverse=True, 
                                                 return_counts=True)
    count_per_row = counts[idx]
    data = data[count_per_row >= th]
    return data
    
def remove_infrequent_places_df(df, th=5):
    place_counts = df.place_id.value_counts()
    mask = (place_counts[df.place_id.values] >= th).values
    df = df[mask]
    return df

def prepare_data(datapath, val_start_day, train_columns, test_columns, th=5):
    val_label = None
    print('Loading train data')
    df_train = load_data(datapath + 'train.csv')
    if val_start_day > 0:
        # Create validation data
        df_train, df_test = validation_split(df_train, val_start_day)
        val_label = df_test['place_id'] 
        df_test.drop(['place_id'], axis=1, inplace=True)    
    print('Feature engineering on train')
    df_train.drop(['row_id'], axis=1, inplace=True)
    df_train = remove_infrequent_places_df(df_train, th)
    gc.collect()
    df_train = feature_engineering(df_train)
    df_train = apply_weights(df_train, fw)
    # reorder the columns so the place id is at the end
    train = df_train[train_columns].values
    del df_train
    gc.collect()
    if val_start_day == 0:
        print('Loading test data')
        df_test = load_data(datapath + 'test.csv') 
    print('Feature engineering on test')
    df_test = feature_engineering(df_test)
    df_test = apply_weights(df_test, fw)
    test = df_test[test_columns].values
    del df_test
    gc.collect()
    return train, test, val_label

def apply_weights(df, fw):
    df['accuracy'] *= fw[0]
    df['day_of_year_sin'] *= fw[1]
    df['day_of_year_cos'] *= fw[1]
    df['minute_sin'] *= fw[2]
    df['minute_cos'] *= fw[2]
    df['weekday_sin'] *= fw[3]
    df['weekday_cos'] *= fw[3]
    df.x *= fw[4]
    df.y *= fw[5]
    df['year'] *= fw[6]
    return df

def feature_engineering(df):
    minute =(df["time"]//5)%288
    trig_arrays = time_trig(288)
    df['minute_sin'] = trig_arrays[0][minute]
    df['minute_cos'] = trig_arrays[1][minute]
    del minute
    day = (df['time']//1440)%365
    trig_arrays = time_trig(365)
    df['day_of_year_sin'] = trig_arrays[0][day]
    df['day_of_year_cos'] = trig_arrays[1][day]
    del day
    weekday = (df['time']//1440)%7
    trig_arrays = time_trig(7)
    df['weekday_sin'] = trig_arrays[0][weekday]
    df['weekday_cos'] = trig_arrays[1][weekday]
    del weekday
    df['year'] = (df['time']//525600).astype(float)
    df.drop(['time'], axis=1, inplace=True)
    df['accuracy'] = np.log10(df['accuracy']).astype(float)
    return df
    
print('Starting...')
start_time = time.time()
# Global variables
datapath = '../input/'
# Change val_start_day to zero to generate predictions
val_start_day = 0 # Day at which to cut validation
th = 5 # Threshold at which to cut places from train
fw = [0.6, 0.32535, 0.56515, 0.2670, 22, 52, 0.51985]

# Defining the size of the grid
x_cuts = 10 # number of cuts along x 
y_cuts = 25 # number of cuts along y
#TODO: More general solution for t_cuts. For now must be 4.
t_cuts = 4 # number of cuts along time. 
x_border_aug = 0.05 # expansion of x border on train 
y_border_aug = 0.015 # expansion of y border on train
time_aug = 2
columns = ['x', 'y', 'minute_sin', 'minute_cos', 'accuracy',
           'day_of_year_sin', 'day_of_year_cos', 'weekday_sin', 
           'weekday_cos', 'year']
train_columns = columns + ['place_id']
test_columns  = columns + ['row_id']

train, test, val_label = prepare_data(datapath, val_start_day,
                                      train_columns, test_columns, th)

elapsed = (time.time() - start_time)
print('Data prepared in:', timedelta(seconds=elapsed))
    
preds = process_grid(train, test, x_cuts, y_cuts, t_cuts,
                     x_border_aug, y_border_aug, time_aug, 
                     fw, th)
elapsed = (time.time() - start_time)
print('Predictions made in:', timedelta(seconds=elapsed))

if val_start_day > 0:
    preds = preds[preds[:, 0] > 0] # only use rows predicted
    labels = val_label.loc[preds[:, 0]].values
    score = mapkprecision(labels, preds[:, 1:])
    print('Final score:', score)
else:
    generate_submission(preds)
elapsed = (time.time() - start_time)
print('Task completed in:', timedelta(seconds=elapsed))