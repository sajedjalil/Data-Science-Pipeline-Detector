import numpy as np
import pandas as pd
import gc

# Data specifications
columns = ['forward_time_delta']
dtypes = {
        'forward_time_delta'    : 'int32'
        }
        
# Training data
print( "Extracting training data...")
training = pd.read_csv( "../input/td-forward-time-deltas-as-csv/td_forward_time_deltas.csv", 
                        nrows=122071523, 
                        usecols=columns, 
                        dtype=dtypes)
                        
# Validation data
print( "Extracting first chunk of validation data...")
valid1 = pd.read_csv( "../input/td-forward-time-deltas-as-csv/td_forward_time_deltas.csv", 
                      skiprows=range(1,144708153), 
                      nrows=7705357, 
                      usecols=columns, 
                      dtype=dtypes)
print( "Extracting second chunk of validation data...")
valid2 = pd.read_csv( "../input/td-forward-time-deltas-as-csv/td_forward_time_deltas.csv", 
                      skiprows=range(1,161974466), 
                      nrows=6291379, 
                      usecols=columns, 
                      dtype=dtypes)
valid2 = pd.concat([valid1, valid2])
del valid1
gc.collect()
print( "Extracting third chunk of validation data...")
valid3 = pd.read_csv( "../input/td-forward-time-deltas-as-csv/td_forward_time_deltas.csv", 
                      skiprows=range(1,174976527), 
                      nrows=6901686, 
                      usecols=columns, 
                      dtype=dtypes)
valid3 = pd.concat([valid2,valid3])
del valid2
gc.collect()
validation = valid3
del valid3
gc.collect()

print( "\nTraining data:")
print( training.shape )
print( training.head() )
print( "Saving training data...")
training.to_pickle('training_forward_deltas.pkl.gz')

validation.reset_index(drop=True,inplace=True)
print( "\nValidation data:")
print( validation.shape )
print( validation.head() )
print( "Saving validation data...")
validation.to_pickle('validation_forward_deltas.pkl.gz')

print("\nDone")