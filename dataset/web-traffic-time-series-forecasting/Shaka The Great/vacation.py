# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

cal = calendar()

dr = pd.date_range(start='2015-07-01' ,end='2017-08-01')

holidays = cal.holidays(start=dr.min(), end=dr.max())

train = pd.read_csv("../input/train_1.csv")

train_flattened = pd.melt(train[list(train.columns[-49:])+['Page']], id_vars='Page', var_name='date', value_name='Visits')

train_flattened['date'] = train_flattened['date'].astype('datetime64[ns]')

train_flattened['vacation'] = ((train_flattened.date.isin(holidays))).astype(float)
print('Average Non Vacation:',train_flattened.loc[train_flattened.vacation==1].Visits.mean()) 
print('Average Non Vacation:',train_flattened.loc[train_flattened.vacation==0].Visits.mean()) 
# Any results you write to the current directory are saved as output.