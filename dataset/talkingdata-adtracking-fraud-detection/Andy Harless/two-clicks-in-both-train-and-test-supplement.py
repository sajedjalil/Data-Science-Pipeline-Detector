import numpy as np
import pandas as pd
import gc

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16'
        }
supp_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time'] 
train = pd.read_csv("../input/train.csv", usecols=supp_columns, skiprows=range(1,144903891), dtype=dtypes)
supp = pd.read_csv("../input/test_supplement.csv", usecols=supp_columns, nrows=40000000, dtype=dtypes)

both = train.merge(supp.drop_duplicates(subset=supp_columns, keep='last'), on=supp_columns, how='inner')
print( both )
