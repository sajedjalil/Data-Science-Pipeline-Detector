# Preprocessing based on Pranav's LGB, as revised by Andy
# Each output subsample contains all positives and a set of negaitves unique to itself

NROWS = 80000000  # Number of rows to read (form end of "train")
NSAMPLES = 70      # Number of negative subsamples to combine with all positives
NSAVE = 24         # Number of combined subsamples to write

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split 
import lightgbm as lgb


def do_count( df, group_cols, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Aggregating by ", group_cols , '...' )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )
    
    

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

print('load train...')
train_cols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
skiprows = 184903891 - NROWS
df = pd.read_csv("../input/talkingdata-adtracking-fraud-detection/train.csv", 
                 skiprows=range(1,skiprows), nrows=NROWS,dtype=dtypes, usecols=train_cols)

import gc

len_train = len(df)

gc.collect()

print('data prep...')

most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
least_freq_hours_in_test_data = [6, 11, 15]

print( "Train info before: ")
print( df.info() )

df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
df.drop(['click_time'], axis=1, inplace=True)
gc.collect()

df['in_test_hh'] = (   3 
                     - 2*df['hour'].isin(  most_freq_hours_in_test_data ) 
                     - 1*df['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')
print( df.info() )

df = do_count( df, ['ip', 'day', 'in_test_hh'], 'nip_day_test_hh', show_max=True, show_agg=True ); 
gc.collect()
df = do_count( df, ['ip', 'day', 'hour'], 'nip_day_hh', 'uint16', show_max=True, show_agg=True ); 
gc.collect()
df = do_count( df, ['ip', 'day', 'os', 'hour'], 'nip_hh_os', 'uint16', show_max=True, show_agg=True ); 
gc.collect()
df = do_count( df, ['ip', 'day', 'app', 'hour'], 'nip_hh_app', 'uint16', show_max=True, show_agg=True ); 
gc.collect()
df = do_count( df, ['ip', 'day', 'app', 'os', 'hour'], 'nip_app_os', 'uint16', show_max=True );
gc.collect()
df = do_count( df, ['app', 'day', 'hour'], 'n_app', 'uint32', show_max=True, show_agg=True ); 
gc.collect()



df.drop( ['ip','day'], axis=1, inplace=True )
gc.collect()
print( df.info() )

df['time_delta'] = pd.read_pickle(
    '../input/talkingdata-time-deltas/train_with_time_deltas.pkl.gz')['time_delta'][-NROWS:].values
    
gc.collect()
print( "Train info after: ")
print( df.info() )

print("vars and data type: ")
df.info()

possubs = np.where(df.is_attributed==1)[0]
negsubs = np.where(df.is_attributed==0)[0]
np.random.shuffle(negsubs)

nnegs = len(negsubs)//NSAMPLES

parts = [ np.concatenate( [possubs, negsubs[i*nnegs:(i+1)*nnegs]] ) for i in range(NSAMPLES) ]

for i in range(NSAVE):
    df.iloc[list(parts[i]),:].to_pickle( 'sample' + str(i) + '.pkl.gz' )
