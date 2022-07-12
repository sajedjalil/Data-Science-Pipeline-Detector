# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.preprocessing import LabelEncoder
import os
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from scipy.stats import rankdata,itemfreq
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from numpy import inf
import pylab as pl

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


datadir = '../input/'
train = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'))
events = pd.read_csv(os.path.join(datadir,'events.csv'),  parse_dates=['timestamp'], index_col='event_id')
appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'))

##### merge data

#get the devices with events
devices_in_events = events['device_id'].unique()
train_in_events = train[train['device_id'].isin(devices_in_events)] #23309
gatrain = train_in_events.reset_index(drop=True)
gatrain = gatrain.set_index('device_id')


# creating two columns to show which train or test set row a particular device_id belongs to.
gatrain['trainrow'] = np.arange(gatrain.shape[0])

appencoder = LabelEncoder().fit(appevents.app_id)
appevents['app'] = appencoder.transform(appevents.app_id)
napps = len(appencoder.classes_)
appevents_Merged = appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)

installedApps_PerDevice = pd.DataFrame([],columns=['device_id','app'])

temp_DeviceIds = list()

eventIds = appevents_Merged['event_id'].unique()
for id in eventIds:
    temp_Device = appevents_Merged.loc[appevents_Merged['event_id'] == id]['device_id'].values[0]
    if temp_Device in temp_DeviceIds:
        #do nothing
        pass
    else:
        installedApps_PerDevice.append(appevents_Merged.loc[appevents_Merged['event_id'] == id])
        temp_DeviceIds.append(temp_Device)

