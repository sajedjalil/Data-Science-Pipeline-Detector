# Create training and validation data sets that mirror the relationship
#   between the training data and the test data

# Based on my script
#   https://www.kaggle.com/aharless/training-and-validation-data
# which is based on Konrad's script
#   https://www.kaggle.com/konradb/validation-set
# and Alexander Firsov's discussion thread
#   https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/51877

import numpy as np
import pandas as pd
import gc

# Data specifications
columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        }
        
# Training data
print( "Extracting training data...")
training = pd.read_csv( "../input/train.csv", 
                        nrows=122071523, 
                        usecols=columns, 
                        dtype=dtypes)
                        
# Validation data
print( "Extracting first chunk of validation data...")
valid1 = pd.read_csv( "../input/train.csv", 
                      skiprows=range(1,144708153), 
                      nrows=7705357, 
                      usecols=columns, 
                      dtype=dtypes)
print( "Extracting second chunk of validation data...")
valid2 = pd.read_csv( "../input/train.csv", 
                      skiprows=range(1,161974466), 
                      nrows=6291379, 
                      usecols=columns, 
                      dtype=dtypes)
valid2 = pd.concat([valid1, valid2])
del valid1
gc.collect()
print( "Extracting third chunk of validation data...")
valid3 = pd.read_csv( "../input/train.csv", 
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
training.to_pickle('training.pkl.gz')

print( "\nValidation data:")
print( validation.shape )
print( validation.head() )
print( "Saving validation data...")
validation.to_pickle('validation.pkl.gz')

print("\nDone")