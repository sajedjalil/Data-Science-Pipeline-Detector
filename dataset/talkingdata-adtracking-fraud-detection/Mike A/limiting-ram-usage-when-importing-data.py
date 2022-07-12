#INTENT: To limit RAM usage when importing/storing data
#Importing all the data at once took > 16 GB, which is my computer's RAM capacity. I made this script to keep the RAM usage low while loading data.

import numpy as np
import pandas as pd

features = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']
int_features = ['ip', 'app', 'device', 'os', 'channel']
time_features = ['click_time', 'attributed_time']
bool_features = ['is_attributed']

for feature in features:
    print("Loading ", feature)
    #Import data one column at a time
    train_unit = pd.read_csv("../input/train_sample.csv", usecols=[feature]) #Change this from "train_sample" to "train" when you're comfortable!
    
    #Pandas imports the numeric data as int64...the following should downgrade that to uint16, saving ~1GB in RAM for each column
    if feature in int_features:    train_unit = pd.to_numeric(train_unit[feature], downcast='unsigned')
    #Convert time data to datetime data, instead of strings
    elif feature in time_features: train_unit=pd.to_datetime(train_unit[feature])
    #Converts the target variable from int64 to boolean. Can also get away with uint16.
    elif feature in bool_features: train_unit = train_unit[feature].astype('bool')
    
    #Make and append each column's data to a dataframe.
    if feature == 'ip': train = pd.DataFrame(train_unit)
    else: train[feature] = train_unit