# This script creates a cross validation dataset. The training data is only from 2013
# and the testing data is only from 2014. The same users are in the training and
# testing datasets.

import pandas as pd
DATE_FEATURES = ['date_time', 'srch_ci', 'srch_co']
TRAIN_ONLY = ['is_booking', 'cnt']
TARGET = 'hotel_cluster'
# Set NROWS to None to load everything.
NROWS = 1000

def has_year(df, year):
    return (df['date_time'].dt.year == year).any()

def has_booking_in_year(df, year):
    return ((df['date_time'].dt.year == year) & (df['is_booking'] == 1)).any()

def load_cv():
    try:
        df_2013 = pd.read_csv('../input/train_2013.csv', parse_dates=DATE_FEATURES)
        bookings_2014 = pd.read_csv('data/test_2014.csv', parse_dates=DATE_FEATURES)
    # OSError is because of the Kaggle infrastructure.
    # Set to IOError for python 2 and FileNotFoundError for python 3.
    except OSError:
        print('CV not found. Building CV datasets.')
        df = pd.read_csv('../input/train.csv', parse_dates=DATE_FEATURES, nrows=NROWS)
        pd.to_datetime(df['date_time'])
        grouped = df.groupby('user_id')
        # We want users who have data in 2013 and bookings in 2014.
        good_users = grouped.filter(lambda x: (has_year(x, 2013) and 
                                               has_booking_in_year(x, 2014)))
        df_2013 = good_users[good_users['date_time'].dt.year == 2013]
        df_2014 = good_users[good_users['date_time'].dt.year == 2014]
        bookings_2014 = df_2014[df_2014['is_booking'] == 1].drop(TRAIN_ONLY, axis=1)
        bookings_2014.insert(0, 'id', range(len(bookings_2014)))
        df_2013.to_csv('train_2013.csv', index=False)
        bookings_2014.to_csv('test_2014.csv', index=False)
    return df_2013, bookings_2014.drop(TARGET, axis=1), bookings_2014[TARGET]
   
if __name__ == '__main__':
    load_cv()
