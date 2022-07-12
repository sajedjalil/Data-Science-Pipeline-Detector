"""
This script provides reusable code for generating lead/lag time
delta features (using epoch time) for an arbitrary choice of lead/lag orders.  

You can use this to generate useful visit time delta features for
this competition,and it should be fairly straightforward to
apply the functions to other datasets as well. Feel free to just
take the output from this kernel as features, they'll match the original
order of train and test. I hope it's helpful!

@author: Joseph Eddy
"""

import numpy as np 
import pandas as pd 

def add_orig_ind_cols(dfs):
    '''
    Add tracker column for original df orders
    '''
    for df in dfs:
        df['orig_ind'] = df.index.values
 
def restore_orig_orders(dfs):
    '''
    Restore original df orders, assumes an 'orig_ind' column
    '''       
    for df in dfs:
        df.sort_values(by='orig_ind', inplace=True)
        df.drop(['orig_ind'], axis=1, inplace=True)
    
def add_grouped_time_delta_features(df, time_col, group_cols, shifts):    
    '''
    For epoch time, compute deltas with the specified shift on sequences
    aggregated by group_cols, return df with new columns
    '''
    
    # sort by time
    df = df.sort_values(by=time_col)
    
    for shift in shifts:
        feat_name = '_'.join(group_cols) + ('_delta_shift_%d' % shift) 
        df[feat_name] = (df.groupby(group_cols)[time_col].shift(shift) - df[time_col]).astype(np.float32).fillna(-1) * -1 * np.sign(shift) 
    return df

read_cols = ['fullVisitorId', 'visitStartTime']  
read_types = {'fullVisitorId': 'str'}

X_train = pd.read_csv('../input/train.csv', usecols=read_cols, dtype=read_types) 
X_test = pd.read_csv('../input/test.csv', usecols=read_cols, dtype=read_types)

# Track original df order to restore at end of feature engineering
add_orig_ind_cols([X_train, X_test])

### Visitor time delta features: lags and leads to order 3
###
print('Extracting time delta features features...\n')

lags = [x for x in range(-3,4) if x != 0]
X_train = add_grouped_time_delta_features(X_train, 'visitStartTime', ['fullVisitorId'], lags)
X_test = add_grouped_time_delta_features(X_test, 'visitStartTime', ['fullVisitorId'], lags)
                                         
# Restore original df order and save new feature outputs
restore_orig_orders([X_train, X_test])

print('Saving feature output...')
X_train.drop('visitStartTime', axis=1).to_csv('train_delta_feats.csv',index=False)
X_test.drop('visitStartTime', axis=1).to_csv('test_delta_feats.csv',index=False)
