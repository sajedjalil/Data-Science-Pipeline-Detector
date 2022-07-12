import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter
import datetime

train = pd.DataFrame()
chunk_size = 10000000
j = 0

for train_chunk in pd.read_csv('../input/train.csv', chunksize=chunk_size):
    train_chunk = train_chunk[train_chunk['is_booking']==1]
    train_chunk = train_chunk.drop(['channel', 'user_location_city', 'user_id', 'is_booking',
                              'orig_destination_distance', 'is_mobile', 'cnt',
                             'user_location_region', 'srch_rm_cnt', 'srch_adults_cnt',
                             'user_location_country', 'srch_destination_id',
                             'srch_children_cnt', 'posa_continent', 'hotel_continent'],
                         axis=1)
    
    train_chunk['date_time'] = pd.to_datetime(train_chunk['date_time'], errors='coerce').map(lambda x: x.date())
    train_chunk['srch_ci'] = pd.to_datetime(train_chunk['srch_ci'], errors='coerce').map(lambda x: x.date())
    train_chunk['srch_ci'][pd.isnull(train_chunk['srch_ci'])] = train_chunk['date_time'][pd.isnull(train_chunk['srch_ci'])]
    train_chunk['srch_co'] = pd.to_datetime(train_chunk['srch_co'], errors='coerce').map(lambda x: x.date())
    train_chunk['srch_co'][pd.isnull(train_chunk['srch_co'])] = train_chunk['date_time'][pd.isnull(train_chunk['srch_co'])]
    train_chunk['srch_ci'] = (train_chunk['srch_ci']-train_chunk['date_time']).map(lambda x: x.days)
    train_chunk['srch_co'] = (train_chunk['srch_co']-train_chunk['date_time']).map(lambda x: x.days)
    train_chunk = train_chunk.drop(['date_time'], axis=1)
    train = train.append(train_chunk)
    j+=1
    print('{} rows of train data processed.'.format(j*chunk_size))
hotel_cluster = train['hotel_cluster']
train = train.drop(['hotel_cluster'], axis=1)

test = pd.read_csv('../input/test.csv')
test = test.drop(['channel', 'user_location_city', 'user_id',
                              'orig_destination_distance', 'is_mobile', 
                             'user_location_region', 'srch_rm_cnt', 'srch_adults_cnt',
                             'user_location_country', 'srch_destination_id',
                             'srch_children_cnt', 'posa_continent', 'hotel_continent'],
                         axis=1)
test['date_time'] = pd.to_datetime(test['date_time'], errors='coerce').map(lambda x: x.date())
test['srch_ci'] = pd.to_datetime(test['srch_ci'], errors='coerce').map(lambda x: x.date())
test['srch_ci'][pd.isnull(test['srch_ci'])] = test['date_time'][pd.isnull(test['srch_ci'])]
test['srch_co'] = pd.to_datetime(test['srch_co'], errors='coerce').map(lambda x: x.date())
test['srch_co'][pd.isnull(test['srch_co'])] = test['date_time'][pd.isnull(test['srch_co'])]
test['srch_ci'] = (test['srch_ci']-test['date_time']).map(lambda x: x.days)
test['srch_co'] = (test['srch_co']-test['date_time']).map(lambda x: x.days)

test = test.drop(['date_time'], axis=1)
test.shape
