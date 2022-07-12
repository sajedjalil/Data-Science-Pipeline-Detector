# Preprocessing based on Pranav's LGB, as revised by Andy
# Each output subsample contains all positives and a set of negaitves unique to itself

NROWS = 80000000  # Number of rows to read (form end of training data)
NSAMPLES = 70      # Number of negative subsamples to combine with all positives
NSAVE = 24         # Number of combined subsamples to write
ENDROW = 122071523 # Rows for training data used during validation
TOTREC = 184903890 # Total rows in original train file

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split 
import lightgbm as lgb
import gc

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
    

print('load train...')
df = pd.read_pickle('../input/training-and-validation-data-pickle/training.pkl.gz')[-NROWS:]

gc.collect()

print('data prep...')


most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
least_freq_hours_in_test_data = [6, 11, 15]

df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
df.drop(['click_time'], axis=1, inplace=True)
gc.collect()

df['in_test_hh'] = (   3 
                     - 2*df['hour'].isin(  most_freq_hours_in_test_data ) 
                     - 1*df['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')

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

endat = TOTREC - ENDROW
df['time_delta'] = pd.read_pickle(
    '../input/just-talkingdata-time-deltas/narrow_train_with_time_deltas.pkl.gz')['time_delta'][-endat-NROWS:-endat]

possubs = np.where(df.is_attributed==1)[0]
negsubs = np.where(df.is_attributed==0)[0]
np.random.shuffle(negsubs)

nnegs = len(negsubs)//NSAMPLES

parts = [ np.concatenate( [possubs, negsubs[i*nnegs:(i+1)*nnegs]] ) for i in range(NSAMPLES) ]

for i in range(NSAVE):
    np.random.shuffle( parts[i] )
    df.iloc[list(parts[i]),:].to_pickle( 'train_sample' + str(i) + '.pkl.gz' )

