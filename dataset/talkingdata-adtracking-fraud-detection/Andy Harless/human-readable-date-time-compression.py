# This script recodes TalkingData click_time to a dhhmmss format (using only one digit
# of date number, since the date range in the training data is only 6 to 9) and uses
# the resulting series to identify the record sequence number for certain times
# (the times that I used to extract training and validation data in my pickle kernel)
#   https://www.kaggle.com/aharless/training-and-validation-data-pickle

import numpy as np
import pandas as pd
import gc

times = pd.read_csv('../input/train.csv', usecols=['click_time'])
days = times.click_time.str[8:10].astype('uint8')
hours = times.click_time.str[11:13].astype('uint8')
minutes = times.click_time.str[14:16].astype('uint8')
seconds = times.click_time.str[17:19].astype('uint8')
times.shape, days.shape, hours.shape, minutes.shape, seconds.shape
del times
gc.collect()

ms = (minutes.astype('uint16')*100 + seconds).astype('uint16')
del minutes, seconds
gc.collect()

hms = (hours.astype('uint32')*10000 + ms).astype('uint32')
del hours, ms
gc.collect()

dhms = (days.astype('uint32')*1000000 + hms).astype('uint32')
del days, hms
gc.collect()

critical_times = [8160001,9040000,9060001,9090000,9110001,9130000,9150001]
print('\nRecords for critical times (dhhmmss):')
for dh in critical_times:
    print(dh, dhms[dhms>=dh].index[0])

print('\nOffsets:')    
print( dhms[dhms>=9060001].index[0] - dhms[dhms>=9040000].index[0] )
print( dhms[dhms>=9110001].index[0] - dhms[dhms>=9090000].index[0] )
print( dhms[dhms>=9150001].index[0] - dhms[dhms>=9130000].index[0] )

