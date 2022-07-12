# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
f = open("../input/train.csv", "r")
line = f.readline()
print(line)

train = pd.read_csv('../input/train.csv'
                    #,dtype={'is_booking':bool,'srch_destination_id':np.int32, 'hotel_cluster':np.int32}
                    #,usecols=['srch_destination_id','is_booking','hotel_cluster']
                    ,chunksize=1000000
                    #,nrows=10000
                   )
train.info()
aggs = []
print('-'*38)
for chunk in train:
    agg = chunk.groupby(['srch_destination_id',
                         'hotel_cluster'])['is_booking'].agg(['sum','count'])
    agg.reset_index(inplace=True)
    aggs.append(agg)
    print('.',end='')
print('')
aggs = pd.concat(aggs, axis=0)
aggs.head()
train = pd.read_csv('../input/train.csv',chunksize=100000)
train
from matplotlib import pyplot as plt
plt.scatter(aggs['sum'][1:100],aggs['count'][1:100])
fit = np.polyfit(aggs['sum'][1:100],aggs['count'][1:100],1)
fit
plt.plot(x,y, 'yo', x, fit_fn(x), '--k')
plt.xlim(0, 5)
plt.ylim(0, 12)
plt.scatter(train['orig_destination_distance'],train['hotel_cluster'])
trainbooking = train[train['is_booking']==1].head()
plt.scatter(trainbooking['orig_destination_distance'],trainbooking['hotel_cluster'])
trainbooking
