# coding: utf-8
__author__ = 'Ravi: https://kaggle.com/company'

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import time
from datetime import timedelta
import gc

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

def process_one_cell(cell_train, cell_test, fw, th, n_neighbors):
    
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
                            n_jobs=2, leaf_size=15)
    clf.fit(X, y)
    y_pred = clf.predict_proba(cell_test)
    y_pred_labels = np.argsort(y_pred, axis=1)[:,:-4:-1]
    pred_labels = clf.classes_[y_pred_labels]
    cell_pred = np.column_stack((row_ids, pred_labels)).astype(np.int64) 
    
    return cell_pred
    
def calculate_distance(distances):
    return distances ** -2.225
    
# Generate a dictionary of the time limits so it doesn't have to be 
# recalculated each loop
def create_time_dict(t_cuts, time_factor, time_aug):
    
    t_slice = 24 / t_cuts
    time_dict = dict()
    for t in range(t_cuts):
        
        t_min = 2 * np.pi * (t * t_slice * 12 / 288)
        t_max = 2 * np.pi * (((t + 1) * t_slice * 12 - 1) / 288)
        sin_t_start = np.round(np.sin(t_min)+1, 4) * time_factor
        sin_t_stop = np.round(np.sin(t_max)+1, 4) * time_factor
        cos_t_start = np.round(np.cos(t_min)+1, 4) * time_factor
        cos_t_stop = np.round(np.cos(t_max)+1, 4) * time_factor
        #print(t, (sin_t_start, sin_t_stop, cos_t_start, cos_t_stop))
        sin_t_min = min((sin_t_start, sin_t_stop))
        sin_t_max = max((sin_t_start, sin_t_stop))
        cos_t_min = min((cos_t_start, cos_t_stop))
        cos_t_max = max((cos_t_start, cos_t_stop))

        time_dict[t] = [sin_t_min, sin_t_max, cos_t_min, cos_t_max]
        t_min = 2 * np.pi * ((t * t_slice - time_aug) * 12 / 288)
        t_max = 2 * np.pi * ((((t + 1) * t_slice + time_aug)* 12 - 1) / 288)
        sin_t_start = np.round(np.sin(t_min)+1, 4) * time_factor
        sin_t_stop = np.round(np.sin(t_max)+1, 4) * time_factor
        cos_t_start = np.round(np.cos(t_min)+1, 4) * time_factor
        cos_t_stop = np.round(np.cos(t_max)+1, 4) * time_factor
        sin_t_min = min((sin_t_start, sin_t_stop, sin_t_min))
        sin_t_max = max((sin_t_start, sin_t_stop, sin_t_max))
        cos_t_min = min((cos_t_start, cos_t_stop, cos_t_min))
        cos_t_max = max((cos_t_start, cos_t_stop, cos_t_max))
        time_dict[t] += [sin_t_min, sin_t_max, cos_t_min, cos_t_max]
        
    return time_dict

def apply_mask(data, feature, mask_min, mask_max):
    mask = (data[:, feature] >= mask_min)
    mask = mask & (data[:, feature] < mask_max)      
    return data[mask]    

def process_grid(train, test, x_cuts, y_cuts, t_cuts,
                 x_border_aug, y_border_aug, time_aug, fw, th, n_neighbors):
    preds_list = []
    x_slice = train[:, 0].max() / x_cuts
    y_slice = train[:, 1].max() / y_cuts
    time_max = train[:, 2].max()
    time_factor = time_max / 2
    time_dict = create_time_dict(t_cuts, time_factor, time_aug)

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
                                             fw, th, n_neighbors)
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
        rows = ['']*preds.shape[0]
        for num in range(preds.shape[0]):
            rows[num]='%d,%d %d %d\n' % (preds[num,0],preds[num,1],preds[num,2],preds[num,3])
        out.writelines(rows)

def validation_split(df, val_start_day):
    day = df['time']//1440
    df_val = df.loc[(day>=val_start_day)].copy()
    df = df.loc[(day<val_start_day)].copy()
    return df, df_val

def remove_infrequent_places(data, th=5):
    places, counts = np.unique(data[:, -1], return_counts=True)
    places = places[counts >= th]
    data = data[np.in1d(data[:, -1], places)]
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
    minute = 2*np.pi*((df["time"]//5)%288)/288
    df['minute_sin'] = (np.sin(minute)+1).round(4)
    df['minute_cos'] = (np.cos(minute)+1).round(4)
    del minute
    day = 2*np.pi*((df['time']//1440)%365)/365
    df['day_of_year_sin'] = (np.sin(day)+1).round(4)
    df['day_of_year_cos'] = (np.cos(day)+1).round(4)
    del day
    weekday = 2*np.pi*((df['time']//1440)%7)/7
    df['weekday_sin'] = (np.sin(weekday)+1).round(4)
    df['weekday_cos'] = (np.cos(weekday)+1).round(4)
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
#fw = [0.6, 0.32535, 0.56515, 0.2670, 22, 52, 0.51985]
fw = [0.61,0.32435, 0.56525, 0.2670, 22, 52, 0.51885]

# Defining the size of the grid
x_cuts = 10 # number of cuts along x 
y_cuts = 25 # number of cuts along y
#TODO: More general solution for t_cuts. For now must be 4.
t_cuts = 4 # number of cuts along time. 
x_border_aug = 0.05 # expansion of x border on train 
y_border_aug = 0.02 # expansion of y border on train
time_aug = 2
n_neighbors = 32
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
                     fw, th, n_neighbors)
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
