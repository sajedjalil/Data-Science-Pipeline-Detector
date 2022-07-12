# This script offers separate parquet files of the 8th and 9th to be used as trainind and testing sets respectively, 

# The goal is to recreate approximatively the conditions of the contest, shifted by a day. It is meant to speed up quick tuning of models.

# The data has been truncated to show only the range of the features appearing in the test data, and the 9th has its timerange reduced as well.

#

# Use pd.read_parquet(file_name).reset_index(drop=True) to read them. It is MUCH faster than compressed csv.

# The reset_index is meant to save space, turning an INT64 index into a Range index.



# Beware, IP is int64 instead of uint32. This is for reason of space for the output. to get the proper dtype, use version='2.0' in the to_parquet call





import numpy as np

import pandas as pd

import os

import gc



        

with open('../input/test.csv','rb') as File:

    data=(pd.read_csv(File,

                        parse_dates=['click_time'],

                        infer_datetime_format=True,

                        usecols=['click_time']

                        )

            .click_time

            )

            

test_times=data.iloc[[0,-1]].values

del(data)

gc.collect()



with open('../input/train.csv','rb') as File:

    data=(pd.read_csv(File,

                        parse_dates=['click_time'],

                        infer_datetime_format=True,

                        usecols=['click_time'])

            .reset_index()

            .set_index('click_time')

            )

dt=pd.Timedelta('1 days')



last_day_index=data[test_times[0]-dt:test_times[1]-dt].iloc[[0,-1],0].values

last_day_skip=range(1,last_day_index[0])

last_day_nrows=last_day_index[1]-last_day_index[0]



ante_day_index=data[test_times[0]-2*dt:test_times[1]-2*dt].iloc[[0,-1],0].values

ante_day_skip=range(1,ante_day_index[0])

ante_day_nrows=ante_day_index[1]-ante_day_index[0]



del(data,last_day_index,ante_day_index,dt)

gc.collect()



columns = ['ip','app','os','device','channel','click_time','is_attributed']     

types = {

        'ip':np.uint32,

        'app': np.uint16,

        'os': np.uint16,

        'device': np.uint16,

        'channel':np.uint16,

        'is_attributed':'bool'

        }



with open('../input/train.csv','rb') as File:

    (pd.read_csv(File,

                parse_dates=['click_time'],

                index_col='click_time',

                infer_datetime_format=True,

                skiprows=ante_day_skip,

                nrows=ante_day_nrows,

                usecols=columns,

                dtype=types,

                engine='c',

                sep=',')

        .tz_localize('UTC')

        .tz_convert('Asia/Shanghai')

        .reset_index()

        #.loc[lambda x: x.ip<126420,:]

        #.loc[lambda x: x.app<521,:]

        #.loc[lambda x: x.channel<498,:]

        #.loc[lambda x: x.os<604,:]

        #.loc[lambda x: x.device<3031,:]

        .to_parquet('ante_day.pqt')

    )

gc.collect()



with open('../input/train.csv','rb') as File:

        (pd.read_csv(File,

                    parse_dates=['click_time'],

                    index_col='click_time',

                    infer_datetime_format=True,

                    skiprows=last_day_skip,

                    nrows=last_day_nrows,

                    usecols=columns,

                    dtype=types,

                    engine='c',

                    sep=',')

            .tz_localize('UTC')

            .tz_convert('Asia/Shanghai')

            .reset_index()

            #.loc[lambda x: x.ip<126420,:]

            #.loc[lambda x: x.app<521,:]

            #.loc[lambda x: x.channel<498,:]

            #.loc[lambda x: x.os<604,:]

            #.loc[lambda x: x.device<3031,:]

            .to_parquet('last_day.pqt')

        )
