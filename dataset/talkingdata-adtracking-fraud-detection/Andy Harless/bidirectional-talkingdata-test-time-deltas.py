
ONE_SECOND = 1000000000    # Number of time units in one second


import pandas as pd
import time
import numpy as np
import psutil
import os
import gc

path = '../input/talkingdata-adtracking-fraud-detection/'
mapping_file_path = '../input/mapping-between-test-supplement-csv-and-test-csv/mapping.csv'

process = psutil.Process(os.getpid())
print('Total memory in use before reading test supplement: ', process.memory_info().rss/(2**30), ' GB\n')    


#######  READ THE DATA  #######

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'click_id'      : 'uint32'
        }
print('\nLoading test supplement...')
test_cols = ['ip','app','device','os', 'channel', 'click_time']
df = pd.read_csv(path+"test_supplement.csv", dtype=dtypes, usecols=test_cols, parse_dates=['click_time'])
df['click_time'] = df.click_time.astype('int64').floordiv(ONE_SECOND).astype('int32')
print('Total memory in use after reading test supplement: ', process.memory_info().rss/(2**30), ' GB\n') 


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
print('Total memory in use after categorizing test supplement: ', process.memory_info().rss/(2**30), ' GB\n')



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
print( 'Calculating time deltas...')
df['catdiff'] = df.category.diff().fillna(1).astype('uint8')
df.drop(['category'],axis=1,inplace=True)
df['time_delta'] = df.click_time.diff().astype('float32')
df.loc[df.catdiff==1,'time_delta'] = np.nan
df['forward_time_delta'] = df.time_delta.shift(-1)

# Make time_deltas integers and re-sort back to the origial order
print( 'Processing final time deltas...')
df = df[['time_delta','forward_time_delta']].fillna(-1).astype('int32')
gc.collect()
df.sort_index(inplace=True)
gc.collect()
print('Total memory in use after diffing: ', process.memory_info().rss/(2**30), ' GB\n')




########  MERGE WITH OFFICIAL TEST DATA  #######

print( 'Mapping to official click_id values for test data...')
df = pd.concat([df, pd.read_csv(path+"test_supplement.csv", dtype=dtypes, usecols=['click_id'])], axis=1)
mapping = pd.read_csv(mapping_file_path, dtype={'click_id': 'int32','old_click_id': 'int32'})
df.rename(columns={'click_id': 'old_click_id'}, inplace=True)
df = pd.merge(df, mapping, on=['old_click_id'], how='right')
df.drop(['old_click_id'], axis=1, inplace=True)
print( 'Missing IDs: ', df['click_id'].isnull().sum() )
print( 'Shape of test file: ', df.shape )
print( df.head() )
df['click_id'] = df['click_id'].astype(np.int32)

print( 'Putting records in order of official test file...')
print( 'Before:')
print('shape: ', df.shape)
print(df.head(10))
click_ids_in_file_order = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=['click_id'])
df = pd.merge(click_ids_in_file_order, df, how='left', on='click_id')
print( 'After:')
print('shape: ', df.shape)
print(df.head(10))



#######  WRITE PICKLE FILE  #######

# Write to disk
print('\nWriting...')
df.to_pickle('bidirectional_test_time_deltas.pkl.gz')
print( 'Done')