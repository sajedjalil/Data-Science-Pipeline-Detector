__author__ = 'Nick Sarris (ngs5st)'

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

def create_features(train, test):

    train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
    test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
    train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)
    train['store_and_fwd_flag'] = 1 * (train.store_and_fwd_flag.values == 'Y')
    test['store_and_fwd_flag'] = 1 * (test.store_and_fwd_flag.values == 'Y')

    train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date
    train.loc[:, 'pickup_weekday'] = train['pickup_datetime'].dt.weekday
    train.loc[:, 'pickup_day'] = train['pickup_datetime'].dt.day
    train.loc[:, 'pickup_month'] = train['pickup_datetime'].dt.month
    train.loc[:, 'pickup_hour'] = train['pickup_datetime'].dt.hour
    train.loc[:, 'pickup_minute'] = train['pickup_datetime'].dt.minute
    train.loc[:, 'pickup_dt'] = (train['pickup_datetime'] - train['pickup_datetime'].min()).map(
        lambda x: x.total_seconds())

    test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date
    test.loc[:, 'pickup_weekday'] = test['pickup_datetime'].dt.weekday
    test.loc[:, 'pickup_day'] = test['pickup_datetime'].dt.day
    test.loc[:, 'pickup_month'] = test['pickup_datetime'].dt.month
    test.loc[:, 'pickup_hour'] = test['pickup_datetime'].dt.hour
    test.loc[:, 'pickup_minute'] = test['pickup_datetime'].dt.minute
    test.loc[:, 'pickup_dt'] = (test['pickup_datetime'] - train['pickup_datetime'].min()).map(
        lambda x: x.total_seconds())

    train['distance_haversine'] = haversine_array(
        train['pickup_latitude'].values, train['pickup_longitude'].values,
        train['dropoff_latitude'].values, train['dropoff_longitude'].values)

    train['distance_dummy_manhattan'] = dummy_manhattan_distance(
        train['pickup_latitude'].values, train['pickup_longitude'].values,
        train['dropoff_latitude'].values, train['dropoff_longitude'].values)

    test['distance_haversine'] = haversine_array(
        test['pickup_latitude'].values, test['pickup_longitude'].values,
        test['dropoff_latitude'].values, test['dropoff_longitude'].values)

    test['distance_dummy_manhattan'] = dummy_manhattan_distance(
        test['pickup_latitude'].values, test['pickup_longitude'].values,
        test['dropoff_latitude'].values, test['dropoff_longitude'].values)

    train['avg_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']
    train['avg_speed_m'] = 1000 * train['distance_dummy_manhattan'] / train['trip_duration']

    train['center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2
    train['center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2
    test['center_latitude'] = (test['pickup_latitude'].values + test['dropoff_latitude'].values) / 2
    test['center_longitude'] = (test['pickup_longitude'].values + test['dropoff_longitude'].values) / 2

    train['pickup_lat_bin'] = np.round(train['pickup_latitude'], 2)
    train['pickup_long_bin'] = np.round(train['pickup_longitude'], 2)
    train['center_lat_bin'] = np.round(train['center_latitude'], 2)
    train['center_long_bin'] = np.round(train['center_longitude'], 2)
    train['pickup_dt_bin'] = (train['pickup_dt'] // (3 * 3600))
    test['pickup_lat_bin'] = np.round(test['pickup_latitude'], 2)
    test['pickup_long_bin'] = np.round(test['pickup_longitude'], 2)
    test['center_lat_bin'] = np.round(test['center_latitude'], 2)
    test['center_long_bin'] = np.round(test['center_longitude'], 2)
    test['pickup_dt_bin'] = (test['pickup_dt'] // (3 * 3600))

    train.loc[:, 'direction'] = bearing_array(
        train['pickup_latitude'].values, train['pickup_longitude'].values,
        train['dropoff_latitude'].values, train['dropoff_longitude'].values)

    test.loc[:, 'direction'] = bearing_array(
        test['pickup_latitude'].values, test['pickup_longitude'].values,
        test['dropoff_latitude'].values, test['dropoff_longitude'].values)

    full = pd.concat([train, test]).reset_index(drop=True)
    coords = np.vstack((full[['pickup_latitude', 'pickup_longitude']],
                        full[['dropoff_latitude', 'dropoff_longitude']]))

    pca = PCA().fit(coords)
    train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]
    train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 1]
    train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
    test['pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]
    test['pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]
    test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

    train['pca_manhattan'] = np.abs(train['dropoff_pca1'] - train['pickup_pca1']) + \
                             np.abs(train['dropoff_pca0'] - train['pickup_pca0'])

    test['pca_manhattan'] = np.abs(test['dropoff_pca1'] - test['pickup_pca1']) + \
                            np.abs(test['dropoff_pca0'] - test['pickup_pca0'])

    train['direction_ns'] = (train.pickup_latitude > train.dropoff_latitude) * 1 + 1
    indices = train[(train.pickup_latitude == train.dropoff_longitude) & (train.pickup_latitude != 0)].index
    train.loc[indices, 'direction_ns'] = 0

    train['direction_ew'] = (train.pickup_longitude > train.dropoff_longitude) * 1 + 1
    indices = train[(train.pickup_longitude == train.dropoff_longitude) & (train.pickup_longitude != 0)].index
    train.loc[indices, 'direction_ew'] = 0

    test['direction_ns'] = (test.pickup_latitude > test.dropoff_latitude) * 1 + 1
    indices = test[(test.pickup_latitude == test.dropoff_longitude) & (test.pickup_latitude != 0)].index
    test.loc[indices, 'direction_ns'] = 0

    test['direction_ew'] = (test.pickup_longitude > test.dropoff_longitude) * 1 + 1
    indices = test[(test.pickup_longitude == test.dropoff_longitude) & (test.pickup_longitude != 0)].index
    test.loc[indices, 'direction_ew'] = 0

    cols_to_drop = ['id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'check_trip_duration',
                    'pickup_date', 'avg_speed_h', 'avg_speed_m', 'pickup_lat_bin', 'pickup_long_bin',
                    'center_lat_bin', 'center_long_bin', 'pickup_dt_bin']

    features = [f for f in train.columns if f not in cols_to_drop]

    train_x = train[features]
    labels = np.log(train['trip_duration'].values + 1)
    test_x = test[features]

    for f in train_x.columns:
        if train_x[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_x[f].values))
            train_x[f] = lbl.transform(list(train_x[f].values))
            test_x[f] = lbl.transform(list(test_x[f].values))

    return train_x, labels, test_x

def nn_model():

    model = Sequential()
    model.add(Dense(units=100, input_dim=25, kernel_initializer='uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.6))

    model.add(Dense(units=40, kernel_initializer='uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.4))
    
    model.add(Dense(units=20, kernel_initializer='uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.3))

    model.add(Dense(units=1, kernel_initializer='uniform', activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model

def main():

    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    train, labels, test = create_features(train, test)

    tr_te = pd.concat([train, test])\
        .reset_index(drop=True)

    ntrain = train.shape[0]
    ntest = test.shape[0]

    scaler = StandardScaler()
    tr_te = scaler.fit_transform(tr_te)

    train = tr_te[:ntrain, :]
    test = tr_te[ntrain:, :]

    print('Dim train', train.shape)
    print('Dim test', test.shape)

    model = nn_model()
    model.fit(train, labels, batch_size=128, epochs=10)
    preds = model.predict(test)[:, 0]

    submit_test = pd.read_csv('../input/test.csv')
    submit_test['trip_duration'] = np.exp(preds) + 1
    submit_test[['id', 'trip_duration']].to_csv('neural_submission.csv', index=False)


if __name__ == '__main__':
    main()