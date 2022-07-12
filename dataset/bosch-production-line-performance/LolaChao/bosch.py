# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
train_date_part = pd.read_csv('../input/train_date.csv', nrows=10000)
print(train_date_part.shape)# (10000,1157)
print(1.0 * train_date_part.count().sum() / train_date_part.size)
print(train_date_part[:2])

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))#display all the doc

# Any results you write to the current directory are saved as output.
# Let's check the min and max times for each station
def get_station_times(dates, withId=False):
    times = []
    cols = list(dates.columns)
    
    if 'Id' in cols:
        cols.remove('Id')
    
    for feature_name in cols:
        if withId:
            df = dates[['Id', feature_name]].copy()
            df.columns = ['Id', 'time']
        else:
            df = dates[[feature_name]].copy()
            df.columns = ['time']
        df['station'] = feature_name.split('_')[1][1:]
        df = df.dropna()
        times.append(df)
    
    return pd.concat(times)
    
    

station_times = get_station_times(train_date_part, withId=True).sort_values(by=['Id', 'station'])
print(station_times[:500])
print(station_times.shape)
min_station_times = station_times.groupby(['Id', 'station']).min()['time']
max_station_times = station_times.groupby(['Id', 'station']).max()['time']
print(np.mean(1. * (min_station_times == max_station_times)))