# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.signal
    
def calculate_time(minutes):
    time_of_day = [4,4,4,4,4,4,0,0,0,0,0,1,1,1,1,2,2,2,3,3,3,3,4,4]
    return ((minutes%(24*60*7))//(24*60))*5+time_of_day[(minutes%(24*60))//60]

train = pd.read_csv('../input/train.csv')

n,m = np.shape(train)
print(str(train.shape))
time = train['time']
time_of_day = [4,4,4,4,4,4,0,0,0,0,0,1,1,1,1,2,2,2,3,3,3,3,4,4]
train['TODOW'] = time.apply(calculate_time)

# Get places with most check-ins
places_by_frequency = train.groupby('place_id')['place_id'].agg('count').sort_values(ascending=False).index.tolist()
n_places_to_analyze = 100
places_by_frequency = places_by_frequency[:n_places_to_analyze]

hist_range = (0, 34)
n_bins = 1

all_autocorrs = np.zeros((n_bins, n_places_to_analyze))
place_n = 0
for place_id in places_by_frequency:
  times = np.squeeze(train[train['place_id']==place_id].as_matrix(columns=['TODOW']))
  n_events = times.size
  n_samples = n_events*n_events # We are still randomly choosing timestamps, but this should give good coverage
  hist_vals, bin_edges = np.histogram(np.random.choice(times, size=n_samples, replace=True) - \
                                    np.random.choice(times, size=n_samples, replace=True), bins=n_bins,
                                    range=hist_range)
  all_autocorrs[:, place_n] = hist_vals
  place_n += 1

plt.plot(bin_edges[:-1], all_autocorrs)
