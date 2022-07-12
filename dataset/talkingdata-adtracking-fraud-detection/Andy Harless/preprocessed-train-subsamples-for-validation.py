# Preprocessing based on Pranav's LGB, as revised by Andy
# Each output subsample contains all positives and a set of negaitves unique to itself

NROWS = 100000000  # Number of rows to read (form end of training data)
NSAMPLES = 70      # Number of negative subsamples to combine with all positives
NSAVE = 35         # Number of combined subsamples to write

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split 
import lightgbm as lgb
import gc

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

print('group by : ip_day_test_hh')
gp = df[['ip', 'day', 'in_test_hh', 'channel']].groupby(by=['ip', 'day',
         'in_test_hh'])[['channel']].count().reset_index().rename(index=str, 
         columns={'channel': 'nip_day_test_hh'})
df = df.merge(gp, on=['ip','day','in_test_hh'], how='left')
del gp
df.drop(['in_test_hh'], axis=1, inplace=True)
print( "nip_day_test_hh max value = ", df.nip_day_test_hh.max() )
df['nip_day_test_hh'] = df['nip_day_test_hh'].astype('uint32')
gc.collect()

print('group by : ip_day_hh')
gp = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 
         'hour'])[['channel']].count().reset_index().rename(index=str, 
         columns={'channel': 'nip_day_hh'})
df = df.merge(gp, on=['ip','day','hour'], how='left')
del gp
print( "nip_day_hh max value = ", df.nip_day_hh.max() )
df['nip_day_hh'] = df['nip_day_hh'].astype('uint16')
gc.collect()

print('group by : ip_hh_os')
gp = df[['ip', 'day', 'os', 'hour', 'channel']].groupby(by=['ip', 'os', 'day',
         'hour'])[['channel']].count().reset_index().rename(index=str, 
         columns={'channel': 'nip_hh_os'})
df = df.merge(gp, on=['ip','os','hour','day'], how='left')
del gp
print( "nip_hh_os max value = ", df.nip_hh_os.max() )
df['nip_hh_os'] = df['nip_hh_os'].astype('uint16')
gc.collect()

print('group by : ip_hh_app')
gp = df[['ip', 'app', 'hour', 'day', 'channel']].groupby(by=['ip', 'app', 'day',
         'hour'])[['channel']].count().reset_index().rename(index=str, 
         columns={'channel': 'nip_hh_app'})
df = df.merge(gp, on=['ip','app','hour','day'], how='left')
del gp
print( "nip_hh_app max value = ", df.nip_hh_app.max() )
df['nip_hh_app'] = df['nip_hh_app'].astype('uint16')
gc.collect()

print('group by : ip_hh_dev')
gp = df[['ip', 'device', 'hour', 'day', 'channel']].groupby(by=['ip', 'device', 'day',
         'hour'])[['channel']].count().reset_index().rename(index=str, 
         columns={'channel': 'nip_hh_dev'})
df = df.merge(gp, on=['ip','device','day','hour'], how='left')
del gp
print( "nip_hh_dev max value = ", df.nip_hh_dev.max() )
df['nip_hh_dev'] = df['nip_hh_dev'].astype('uint32')
gc.collect()

df.drop( ['ip','day'], axis=1, inplace=True )
gc.collect()


possubs = np.where(df.is_attributed==1)[0]
negsubs = np.where(df.is_attributed==0)[0]
np.random.shuffle(negsubs)

nnegs = len(negsubs)//NSAMPLES

parts = [ np.concatenate( [possubs, negsubs[i*nnegs:(i+1)*nnegs]] ) for i in range(NSAMPLES) ]

for i in range(NSAVE):
    np.random.shuffle( parts[i] )
    df.iloc[list(parts[i]),:].to_pickle( 'train_sample' + str(i) + '.pkl.gz' )

