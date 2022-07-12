# Companion to:
#    https://www.kaggle.com/aharless/training-and-validation-data-pickle

# This provides a "validation supplement" to use during training
#    as an analogue to the test data supplement

# A perfect analogue isn't possible given the way the data are divided
#    (since the time periods for training and test_supplement overlap),
#    so this divides the raw train data just after the end 
#    of the analogue training data created by the earlier script

import numpy as np
import pandas as pd
import gc

# Data specifications
columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        }

print( "Extracting validation supplement data...")
columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
valid_supp = pd.read_csv( "../input/train.csv",
                        skiprows=range(1,122071523), 
                        usecols=columns, 
                        dtype=dtypes)
                        
print( "\nValidation supplement data:")
print( valid_supp.shape )
print( valid_supp.head() )
print( "Saving validation supplement data...")
valid_supp.to_pickle('valid_supp.pkl.gz')

print("\nDone")