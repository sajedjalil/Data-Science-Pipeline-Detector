# Find the first occurrence of each IP address in TalkingData data

import numpy as np
import pandas as pd
import gc

print( "\nExtracting training data...")
df = pd.read_csv( "../input/train.csv", usecols=['ip'], dtype='uint32')
print( "Train shape: ", df.shape )
print( df.head() )

print( "\nExtracting test data (full supplement)...")
test = pd.read_csv("../input/test_supplement.csv", usecols=['ip'], dtype='uint32')
print( "Test (full supplement) shape: ", test.shape )
print( test.head() )
df['test'] = False
test['test'] = True
                        
print( "\nCombining data...")
df = df.append(test)
del test
gc.collect()
print( "Combined shape: ", df.shape )
print( df.head() )

print( "\nCalculating IP ordinals (cumulative counts)..." )
counts = df.groupby('ip').cumcount().astype('uint32')
gc.collect()
df['ip_ordinal'] = counts
del counts
gc.collect()
print( "Combined shape: ", df.shape )
print( df.head() )

df.drop(['ip'], axis=1, inplace=True)
gc.collect()

print( "\nWriting train ordinals to disk..." )
df[~df['test']][['ip_ordinal']].to_pickle('train_ordinal.pkl.gz')

df = df[df['test']][['ip_ordinal']]
gc.collect()
print( "\nFull test ordinals shape: ", df.shape )
print( df.head() )

print( "\nExtracting test data (full supplement) again..." )
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16'
        }
supp_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time'] 
supp = pd.read_csv("../input/test_supplement.csv", usecols=supp_columns, dtype=dtypes)
supp['ip_ordinal'] = df.ip_ordinal
del df
gc.collect()
print( "Full test supplement shape: ", supp.shape )
print( supp.head() )

print( "\nExtracting official test data..." )
test_columns = supp_columns + ['click_id']
dtypes['click_id'] = 'uint32'
test = pd.read_csv("../input/test.csv", usecols=test_columns, dtype=dtypes)
print( "Official test shape: ", test.shape )
print( test.head() )

print( "\nMerging ordinals into test data..." )
test = test.merge(supp.drop_duplicates(subset=supp_columns, keep='last'), on=supp_columns, how='left')[['ip_ordinal']]
del supp
gc.collect()
print( "Test ordinals shape: ", test.shape )
print( test.head() )

print( "\nWriting test ordinals to disk..." )
test.to_pickle('test_ordinal.pkl.gz')

print( "Done." )
