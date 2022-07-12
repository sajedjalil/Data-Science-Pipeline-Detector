# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import pandas as pd
import numpy as np
import time
import gc
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool

#check if difference is within last min, hour, 24hrs or earlier
def getpastclicks(diff, index):
    df1.at[index, 'time_diff'] = diff
    if diff.days >= 1:
        df1.at[index, 'clicks_prior_to_24hrsorearlier'] += 1
    else:
        hours, remainder = divmod(diff.seconds, 3600)
        mins, secs = divmod(remainder, 60)
        if hours >= 1 and hours <= 24:
            df1.at[index, 'clicks_in_last_24hrs'] += 1
        else:
            if mins >= 1 and mins <= 60:
                df1.at[index, 'clicks_in_last_onehr'] +=1
            else:
                if secs >= 1 and secs <= 60:
                    df1.at[index, 'clicks_in_last_onemin'] += 1

def iterate_groups(group):
    # get indices in group to a list and iterate over it
    indices_in_group = group.index.tolist()
    # no need to iterate if theres just one element in group
    index_len = len(indices_in_group)
    # if index_len > 1:
    #iterate indices in group to calculate time diff
    index_clktime = group.columns.get_loc('click_time')
    for i in range(0, index_len):
        t1 = group.iat[i, index_clktime]
        for j in range(i + 1, index_len):
            t2index = indices_in_group[j]
            t2 = group.iat[j, index_clktime]
            if t1 == t2:
                df1.at[t2index, 'clicks_in_last_onemin'] += 1
            else:
                diff = t2 - t1
                getpastclicks(diff, t2index)

def get_skiprows(i):
    if i == 1:
        n1 = i
        n2 = i * rows_percore
    else :
        n1 = (i-1) * rows_percore
        n2 = i * rows_percore
    return [x for x in range(n1, n2) if x % k != 0]

###############################################################################

starttime = time.time()

n_cores = multiprocessing.cpu_count()
print("\nnumber of cores " + str(n_cores))

# sampling by reading every kth index
path = "../input/"
file = path + "train.csv"
s = 1000000          # approx sample size

n = sum(1 for l in open(file))       # number of rows in file
print("\nnumber of lines in file " + str(n))
k = round(n/s)
rows_percore = int(n/n_cores)
nth_core = [x for x in range(1, n_cores+1)]
skip_list = []
print("calculating rows to skip")
if __name__ == '__main__':
    pool = Pool(processes = n_cores)
    skip_list = pool.map(get_skiprows, nth_core)
    pool.close()

skip_ids = [item for sublist in skip_list for item in sublist]
n_rows_skipping = len(skip_ids)
print("\nsample size " + str(n - n_rows_skipping - 1))

del skip_list
gc.collect

print("\nreading file...")
# datatypes
dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'bool'
        }

df = pd.read_csv(file, dtype = dtypes, parse_dates = ['click_time', 'attributed_time'], infer_datetime_format = True, skiprows = skip_ids)

del skip_ids
gc.collect

print("time taken loading file - " + str(time.time() - starttime))
print('\nmemory usage - ')
df.info(memory_usage = 'deep')
print(df.memory_usage(deep = True)/1024)
print('\ndataframe head - ')
print(df.head())



# assuming each phone can be identified with combination of cols in groupby_cols, calculate  count of previous clicks
groupby_cols = ['ip', 'device', 'os']
groupby_cols_n_clktime = groupby_cols + ['click_time']

# separate required columns by splitting df
df1 = df[groupby_cols_n_clktime]
df2 = df.drop(columns = groupby_cols_n_clktime)
del df
gc.collect

# add new columns/new features to be added
df1['time_diff'] = 0
df1['clicks_prior_to_24hrsorearlier'] = 0
df1['clicks_in_last_24hrs'] = 0
df1['clicks_in_last_onehr'] = 0
df1['clicks_in_last_onemin'] = 0

# add new cols/features and set their datatypes
newcol_dtypes = {
        'time_diff': 'uint16',
        'clicks_prior_to_24hrsorearlier': 'uint16',
        'clicks_in_last_24hrs': 'uint16',
        'clicks_in_last_onehr': 'uint16',
        'clicks_in_last_onemin': 'uint16'
        }

df1 = df1.astype(dtype = newcol_dtypes) 

# group by groupby_cols, iterate over and add features
print("\nadding features\n")
featuretime = time.time()
# add features without parallelization
df1.groupby(groupby_cols).apply(lambda group: iterate_groups(group) if len(group.index.tolist()) > 1 else group)

print("\nadding additional features\n")
# add day, hour, minute, second from click_time
df1['day_of_week'] = df1['click_time'].dt.dayofweek
df1['hour_of_day'] = df1['click_time'].dt.hour
df1['min_of_hour'] = df1['click_time'].dt.minute
df1['sec_of_min'] = df1['click_time'].dt.second

print("\nadded new features")
print("time taken for adding features " + str(time.time() - featuretime))

# for manual verification
#df1.sort_values(by = groupby_cols, inplace = True)

df = pd.concat([df1, df2], axis=1, join='inner')
df = df.drop(columns = ['time_diff'], axis = 1)

#del df1, df2
#gc.collect

#write dataframe to hdf file
print("\nwriting to file")
df.to_hdf('train_hdf', key = 'train_features', mode = 'w', format = 'table')
#df_hdf = pd.read_hdf('train_hdf_500k_id_device', key = 'train_features', mode = 'r')

print('\ntotal run time ' + str(time.time() - starttime))