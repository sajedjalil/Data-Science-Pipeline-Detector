# Version 7 is revised to retain original row order within categories
# instead of sorting by time.

# This script calculates time differences between successive clicks 
#   within categories of ip-by-app-by-device-by-os-by-channel

# It processes the whole training file, but, due to disk space limits,
#   the output omits the firt 45 million records.  However, the time deltas
#   reported on subsequent records still reflect the differences 
#   from those early records.

# The column "time_delta" is appended to the "train" data frame,
#   which is saved as a pickle file (with "attriuted_time" omited).

# The time_delta for the first occurrence of each category is
#   recorded as NaN, and the subsequent deltas are recorded in seconds.

# This script is meant as a template.  Various variations suggest themselves.
#   For example, one could save time deltas for the whole training file and
#   reconstitute the file after reading it, so as to avoid the disk write limit.
#   Or one could generate time deltas for different categorizations.

# And of course it will be necessary to generate time deltas for
#   the test data as well, ideally using test_supplement and
#   combining it with the training data.  (But processing the combination 
#   is probably not possible in a Kaggle kernel without breaching
#   the memory constraint, so some workaround will be necessary.)


SKIP_ON_OUTPUT = 45000000  # First records won't be written because of limited disk space
ONE_SECOND = 1000000000    # Number of time units in one second


import pandas as pd
import time
import numpy as np
import psutil
import os
import gc

path = '../input/'
process = psutil.Process(os.getpid())
print('Total memory in use before reading train: ', process.memory_info().rss/(2**30), ' GB\n')    


#######  READ THE DATA  #######

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'click_id'      : 'uint32'
        }
print('\nLoading train...')
train_cols = ['ip','app','device','os', 'channel', 'click_time']
df = pd.read_csv(path+"train.csv", dtype=dtypes, usecols=train_cols, parse_dates=['click_time'])
df['click_time'] = df.click_time.astype('int64').floordiv(ONE_SECOND).astype('int32')
print('Total memory in use after reading train: ', process.memory_info().rss/(2**30), ' GB\n') 


#######  GENERATE COMBINED CATEGORY FOR GROUPING  #######

# Collapse all categorical features into a single feature
imax = df.ip.max()
amax = df.app.max()
dmax = df.device.max()
omax = df.os.max()
cmax = df.channel.max()
print( imax, amax, dmax, omax, cmax )
df['category'] = df.ip.astype('int64')
df.drop(['ip'], axis=1, inplace=True)
df['category'] *= amax
df['category'] += df.app
df.drop(['app'], axis=1, inplace=True)
df['category'] *= dmax
df['category'] += df.device
df.drop(['device'], axis=1, inplace=True)
df['category'] *= omax
df['category'] += df.os
df.drop(['os'], axis=1, inplace=True)
df['category'] *= cmax
df['category'] += df.channel
df.drop(['channel'], axis=1, inplace=True)
gc.collect()

# Replace values for combined feature with a group ID, to make it smaller
print('\nGrouping by combined category...')
df['category'] = df.groupby(['category']).ngroup().astype('uint32')
gc.collect()
print('Total memory in use after categorizing train: ', process.memory_info().rss/(2**30), ' GB\n')



#######  SORT BY CATEGORY AND INDEX  #######

# Collapse category and index into a single column
df['category'] = df.category.astype('int64').multiply(2**32).add(df.index.values.astype('int32'))
gc.collect()

# Sort by category+index (leaving each category separate, sorted by index)
print('\nSorting...')
df = df.sort_values(['category'])
gc.collect()

# Retrieve category from combined column
df['category'] = df.category.floordiv(2**32).astype('int32')
gc.collect()
print('Total memory in use after sorting: ', process.memory_info().rss/(2**30), ' GB\n')



#######  GENERATE TIME DELTAS  #######

# Calculate time deltas, and replace first record for each category by NaN
df['catdiff'] = df.category.diff().fillna(1).astype('uint8')
df.drop(['category'],axis=1,inplace=True)
df['time_delta'] = df.click_time.diff().astype('float32')
df.loc[df.catdiff==1,'time_delta'] = np.nan

# Re-sort time_delta back to the origial order
time_delta = df['time_delta'].sort_index()
del df
gc.collect()
print('Total memory in use after diffing: ', process.memory_info().rss/(2**30), ' GB\n')



#######  ADD TIME DELTAS INTO ORIGINAL TRAINING DATA  #######

# Reload training data
print('\nLoading train again...')
df = pd.read_csv(path+"train.csv", dtype=dtypes, usecols=train_cols, parse_dates=['click_time'])
print('Total memory in use after re-reading train: ', process.memory_info().rss/(2**30), ' GB\n')

# Add time_delta to training data
df['time_delta'] = time_delta
print('Total memory in use after adding time deltas: ', process.memory_info().rss/(2**30), ' GB\n')



#######  WRITE PICKLE FILE  #######

# Clean up
del time_delta
df = df.iloc[SKIP_ON_OUTPUT:]
gc.collect()
print('Total memory in use before writing result: ', process.memory_info().rss/(2**30), ' GB\n')

# Write to disk
print('\nWriting...')
df.to_pickle('train_with_time_deltas.pkl.gz')
print( 'Done')