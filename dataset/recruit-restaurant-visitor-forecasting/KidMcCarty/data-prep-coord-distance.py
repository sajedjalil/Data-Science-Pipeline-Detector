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

import sys
import os

import datetime as dt

from math import sin, cos, sqrt, atan2, radians


def get_distance(lat1, lon1, lat2, lon2):
    """
    
    :param lat1: 
    :param lon1: 
    :param lat2: 
    :param lon2: 
    :return: 
    """

    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance


# Functions
def add_date_info(tbl, column):
    """
    
    :param tbl: 
    :param column: 
    :return: 
    """

    #
    pre_len = len(tbl)
    tbl = pd.merge(
        tbl,
        date_info,
        how='left',
        left_on=[column],
        right_on=['calendar_date'],
        suffixes=['', '_infoTable']
    )
    if len(tbl) != pre_len:
        raise Exception("Merge of type \'left\' has added rows unepectedly.")

    # Remove columns added via suffix
    try:
        tbl.drop(labels=[x for x in tbl.columns if '_infoTable' in x],
                 axis=1,
                  inplace=True)
    except ValueError as VE:
        print(VE)
        pass

    # Rename
    tbl.rename(
        columns={'day_of_week': '{}_day_of_week'.format(column),
                 'holiday_flg': '{}_holiday_flg'.format(column)},
        inplace=True
    )

    return tbl


def add_store_info(tbl, store_tbl, column):
    """

        :param tbl: 
        :param column: 
        :return: 
        """

    #
    pre_len = len(tbl)
    tbl = pd.merge(
        tbl,
        store_tbl,
        how='left',
        left_on=[column],
        right_on=[column]
    )
    if len(tbl) != pre_len:
        raise Exception("Merge of type \'left\' has added rows unepectedly.")

    return tbl


def add_store_relation(tbl, column):
    """
    
    :param tbl: 
    :param column: 
    :return: 
    """

    pre_len = len(tbl)
    tbl = pd.merge(
        tbl,
        store_relation,
        how='left',
        on=column
    )
    if pre_len != len(tbl):
        raise Exception("Merge of type \'left\' has added rows unexpectedly")

    return tbl


def nearestcomps(master_table, k, min_dist=None, by_genre=None):
    """
    
    :param by_genre: 
    :return: 
    """

    # Set dict for nearest of not merging back
    nearest_dict = {}

    # Determine which latitude / longitude to keep as final
    master_table = master_table.loc[:, [
        'storeId', 'latitude', 'longitude'
    ]].drop_duplicates(inplace=False)
    master_table['coords_zip'] = list(zip(
        master_table['latitude'], master_table['longitude']
    ))

    # Iterate over each
    for store in list(set(master_table['storeId'])):

        # Separate target from the rest that may be close
        target = master_table.loc[master_table['storeId'] == store, :]
        target_lat = target['coords_zip'].iloc[0][0]
        target_lon = target['coords_zip'].iloc[0][1]
        if by_genre:
            target_genre =target['by_genre']

        # Add column to rest showing distance
        rest = master_table.loc[master_table['storeId'] != store, :]
        rest['distance'] = rest['coords_zip'].apply(
            lambda x: get_distance(
                target_lat, target_lon,
                x[0], x[1]
            )
        )

        # Get min
        rest.sort_values(by=['distance'],
                         ascending=True,
                         inplace=True)
        # Filter to req distance away
        if min_dist:
            rest = rest.loc[rest['distance'] <= min_dist, :]

        # Get nearest k
        rest = rest.head(k)
        nearest_dict[store] = list(rest['storeId'])

    return nearest_dict


# General path for raw files and where to create output directories
proj_path = '../input/'


# -----------------------------------------------------------------------------
# --------------
# Read in all
air_visit = pd.read_csv(proj_path + 'air_visit_data.csv',
                        parse_dates=['visit_date'])
air_reserve = pd.read_csv(proj_path + 'air_reserve.csv',
                          parse_dates=['visit_datetime','reserve_datetime'])
hpg_reserve = pd.read_csv(proj_path + 'hpg_reserve.csv',
                          parse_dates=['visit_datetime','reserve_datetime'])
date_info = pd.read_csv(proj_path + 'date_info.csv',
                        parse_dates=['calendar_date'])
air_store_info = pd.read_csv(proj_path + 'air_store_info.csv')
hpg_store_info = pd.read_csv(proj_path + 'hpg_store_info.csv')
store_relation = pd.read_csv(proj_path + 'store_id_relation.csv')
test = pd.read_csv(proj_path + 'sample_submission.csv')

# --------------
# Add Date info and Store info to all other tables with dates

# Air Visit
air_visit = add_date_info(tbl=air_visit, column='visit_date')
air_visit = add_store_info(tbl=air_visit, store_tbl=air_store_info,
                           column='air_store_id')


# Air Reserve
air_reserve = add_date_info(tbl=air_reserve, column='visit_datetime')
air_reserve = add_date_info(tbl=air_reserve, column='reserve_datetime')
air_reserve = add_store_info(tbl=air_reserve, store_tbl=air_store_info,
                             column='air_store_id')
air_reserve = add_store_relation(tbl=air_reserve, column='air_store_id')

# HPG Reserve
hpg_reserve = add_date_info(tbl=hpg_reserve, column='visit_datetime')
hpg_reserve = add_date_info(tbl=hpg_reserve, column='reserve_datetime')
hpg_reserve = add_store_info(tbl=hpg_reserve, store_tbl=hpg_store_info,
                             column='hpg_store_id')
hpg_reserve = add_store_relation(tbl=hpg_reserve, column='hpg_store_id')

# --------------
# Combine tables using store_relation

# Combine Reserves
reserve_table = pd.merge(
    air_reserve, hpg_reserve, how='outer', on=['air_store_id', 'hpg_store_id'],
    suffixes=['', '_online']
)

# Assign out to master storeId value, default to air if both are present
reserve_table['storeId'] = np.NaN
reserve_table.loc[reserve_table['air_store_id'].isnull(), 'storeId'] = \
    reserve_table['hpg_store_id']
reserve_table.loc[reserve_table['hpg_store_id'].isnull(), 'storeId'] = \
    reserve_table['air_store_id']
reserve_table.loc[((reserve_table['air_store_id'].notnull()) &
                   (reserve_table['hpg_store_id'].notnull())), 'storeId'] = \
    reserve_table['air_store_id']

# Assign out to master coordinates
for coord in ['latitude', 'longitude']:

    # Assert equal coordinates from each source
    # equal_msk = (
    #     (reserve_table[coord].notnull()) &
    #     (reserve_table['{}_online'.format(coord)].notnull()) &
    #     (reserve_table[coord] != reserve_table['{}_online'.format(coord)])
    # )
    # assert sum(equal_msk) == 0

    reserve_table.loc[reserve_table[coord].isnull(), coord] = \
        reserve_table['{}_online'.format(coord)]
    reserve_table.loc[reserve_table[coord].isnull(), coord] = \
        reserve_table['{}_online'.format(coord)]

# Get dictionary of nearestcomps to be references for each storeId
neighbor_dict = nearestcomps(master_table=reserve_table, k=20,
                             min_dist=20, by_genre=False)


# -----------------------------------------------------------------------------
# Feature Engineering
for date_col in ['visit_datetime', 'reserve_datetime']:

    # Handle extracting dates
    reserve_table[date_col.strip('_datetime') + '_date'] = \
        reserve_table[date_col].dt.strftime('%m_%d_%Y')

    # Handle extracting times
    reserve_table[date_col.strip('_datetime') + '_time'] = \
        reserve_table[date_col].dt.strftime('%H_%M_%S')