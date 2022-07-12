import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
import matplotlib.dates 
import datetime 
%matplotlib inline
train = pd.read_csv('../input/train.csv')
store = pd.read_csv('../input/store.csv')
test = pd.read_csv('../input/test.csv') 
print 'training set size', train.shape 
print 'testing set size', test.shape
print 'store info', store.shape

store.head()

train.head()

test.head()


import matplotlib.pyplot as plt 
import matplotlib.dates 
import datetime 
datetimes = [datetime.datetime.strptime(t, "%Y-%m-%d") for t in train.Date]


plotData = matplotlib.dates.date2num(datetimes) 

train = train.join(pd.DataFrame(plotData,columns = ['datetimes']))

def splitTime(x): 
    mysplit = datetime.datetime.strptime(x,  "%Y-%m-%d") 
    return [mysplit.year,mysplit.month,mysplit.day]
train = train.join(pd.DataFrame(train.Date.apply(splitTime).tolist(), columns = ['year','mon','day']))