'''
This script is meant for checking whether ID column is encoded into datetime format or not.
Surprisingly I found a datetime pattern after decoding the ID column for both train and test set.
Hopefully this column will be helpful for time series analysis. Eagerly waiting for opinions. 
Thanks.
-- Sayantan Mukherjee --

'''

import numpy as np 
import pandas as pd 
import datetime
import os

# function for converting hex to datetime format
def hex_to_datetime(string):
    string = ''.join(reversed(string.split()))
    return(datetime.datetime.fromtimestamp(int(string,16)).strftime('%Y-%m-%d %H:%M:%S'))


train = pd.read_csv('../input/train.csv')
test= pd.read_csv('../input/test.csv')


train_id=train['ID'].apply(hex_to_datetime)
test_id=test['ID'].apply(hex_to_datetime)

print ("--------------------------------  START  --------------------------------------")
print ("train id : "+ train_id)
print ("------------------------------------------------------------------------")
print ("test id : "+ test_id)
print ("--------------------------------  STOP  ----------------------------------------")

