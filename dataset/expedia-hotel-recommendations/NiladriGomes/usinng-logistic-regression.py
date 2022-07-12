# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input dataimport scipy
from scipy.optimize import fmin_bfgs
import matplotlib
import matplotlib.pyplot as plot
#from sklearn import preprocessing
#from matplotlib import style
import pylab
import datetime
import re
# files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
df = pd.read_csv('../input/train.csv',usecols = ['is_booking','srch_adults_cnt','srch_destination_id',\
'srch_ci','srch_co','hotel_cluster'],chunksize=1000)

df = pd.concat(df, ignore_index=True)

df = df.groupby(['is_booking']).get_group(1)

dfx = df.ix[:,'hotel_cluster']
ylabel = dfx.value_counts()
ylabel = ylabel.index
y = dfx.as_matrix()


dfx = df.ix[:,'srch_adults_cnt']
x1 = dfx.as_matrix()
mu = np.mean(x1)
s = np.amax(x1)-np.amin(x1)
x1 = (x1 - mu)/s

dfx = df.ix[:,'srch_destination_id']
x2 = dfx.as_matrix()
mu = np.mean(x2)
s = np.amax(x2)-np.amin(x2)
x2 = (x2 - mu)/s


dfx = df.ix[:,'srch_ci']
dfx = pd.to_datetime(dfx)
ci = dfx.dt.year*365 + dfx.dt.month*30 + dfx.dt.day

dfx = df.ix[:,'srch_co']
dfx = pd.to_datetime(dfx)
co = (dfx.dt.year)*365 + (dfx.dt.month)*30 + dfx.dt.day
x3 = co - ci
x3 = x3.as_matrix()
mu = np.mean(x3)

s = np.amax(x3)-np.amin(x3)
x3 = (x3 - mu)/s

x0 = np.ones(len(y))

X = np.vstack((x0,x1,x2,x3))

print(X.shape)                                        
print('Saving X and y values...')
np.savetxt('Xvalue.out',X)
np.savetxt('yvalue.out',y)
print('Done!')

#print(X)
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.