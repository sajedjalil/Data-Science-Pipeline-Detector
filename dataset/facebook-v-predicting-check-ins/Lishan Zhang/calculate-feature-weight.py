from sklearn.preprocessing import LabelEncoder
import math
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import time
from datetime import timedelta
import gc
from sklearn.linear_model import LogisticRegression


# the weight will be dramatically changed based upon the number of neighbors


def load_data(data_name):
    types = {'row_id': np.dtype(np.int32),
             'x': np.dtype(float),
             'y': np.dtype(float),
             'accuracy': np.dtype(np.int16),
             'place_id': np.int64,
             'time': np.dtype(np.int32)}
    df = pd.read_csv(data_name, dtype=types, index_col=0, na_filter=False)
    return df


def calculate_distance(distances):
    return 1.0 / (np.e ** (1 + 1.5 * distances))


def get_a_cell_data(df_train, x_cuts, y_cuts,
                    i, j, th):
    preds_list = []
    x_slice = df_train['x'].max() / x_cuts
    y_slice = df_train['y'].max() / y_cuts

    x_min = x_slice * i
    x_max = x_slice * (i + 1)
    y_min = y_slice * j
    y_max = y_slice * (j + 1)

    mask = (df_train['x'] >= x_min)
    mask = mask & (df_train['x'] < x_max)
    mask = mask & (df_train['y'] >= y_min)
    mask = mask & (df_train['y'] < y_max)
    cell_train = df_train[mask]

    cell_train = remove_infrequent_places(cell_train, th)

    return cell_train


def remove_infrequent_places(df, th=5):
    place_counts = df.place_id.value_counts()
    mask = (place_counts[df.place_id.values] >= th).values
    df = df.loc[mask]
    return df


def prepare_data(datapath):
    df_train = load_data(datapath + 'train.csv')

    print('Feature engineering on train')
    # df_train = remove_inaccurate(df_train)
    df_train = feature_engineering(df_train)
    # df_train = apply_weights(df_train, fw)
    return df_train


def apply_weights(df, fw):
    df['accuracy'] *= fw[0]
    df['day_of_year_sin'] *= fw[1]
    df['day_of_year_cos'] *= fw[1] * 1.6
    df['minute_sin'] *= fw[2] * 1.1
    df['minute_cos'] *= fw[2]
    df['weekday_sin'] *= fw[3] * 1.1
    df['weekday_cos'] *= fw[3]
    df.x *= fw[4]
    df.y *= fw[5]
    df['year'] *= fw[6]
    return df


def feature_engineering(df):
    minute = (df['time'] % (24 * 60)) * 1.0 / (24 * 60) * 2 * np.pi
    df['minute_sin'] = (np.sin(minute) + 1).round(4)
    df['minute_cos'] = (np.cos(minute) + 1).round(4)
    del minute
    day = 2 * np.pi * ((df['time'] // 1440) % 365) / 365
    df['day_of_year_sin'] = (np.sin(day) + 1).round(4)
    df['day_of_year_cos'] = (np.cos(day) + 1).round(4)
    del day
    weekday = 2 * np.pi * ((df['time'] // 1440) % 7) / 7
    df['weekday_sin'] = (np.sin(weekday) + 1).round(4)
    df['weekday_cos'] = (np.cos(weekday) + 1).round(4)
    del weekday
    # df['sin_cos_min']=df['minute_sin']*df['minute_cos']
    df['year'] = (((df['time']) // 525600))
    df.drop(['time'], axis=1, inplace=True)
    df['accuracy'] = np.log10(df['accuracy'])
    return df


def gen_logit_data(df, fw, n_neighbors):
    df_x = df.drop(['place_id'], axis=1)
    df_y = df['place_id']
    logit_x = []
    logit_y = []
    df_train_x = apply_weights(df_x.copy(), fw)

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(df_train_x, df_y)
    dists_list, neighbors_list = clf.kneighbors(df_train_x, n_neighbors)

    print (neighbors_list.shape)

    df_x=df_x.values
    df_y=df_y.values
    for i in range(0, len(neighbors_list)):
        # process a query point
        neighbors = neighbors_list[i]
        for j in range(1, len(neighbors)):
            neighbor_index = neighbors[j]
            new_logit_x = abs(df_x[i] - df_x[neighbor_index])
            new_logit_y = (df_y[i] == df_y[neighbor_index]) + 0
            logit_x.append(new_logit_x)
            logit_y.append(new_logit_y)

    return logit_x, logit_y


print('Starting...')
start_time = time.time()
# Global variables
datapath = '../input/'
th = 5  # Threshold at which to cut places from train

# Defining the size of the grid
x_cuts = 20  # number of cuts along x
y_cuts = 20  # number of cuts along y
n_neighbors = 3

df_train = prepare_data(datapath)
gc.collect()

elapsed = (time.time() - start_time)
print('Data prepared in:', timedelta(seconds=elapsed))

cell_data = get_a_cell_data(df_train, x_cuts, y_cuts,
                            9, 9, th)

fw = [0.6, 0.32935, 0.56515, 0.2670, 22, 52, 0.51785]
logit_x, logit_y = gen_logit_data(cell_data, fw, n_neighbors)
print (np.mean(logit_y))

logit = LogisticRegression()
logit.fit(logit_x, logit_y)

print (logit.coef_)

print('Task completed in:', timedelta(seconds=elapsed))