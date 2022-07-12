# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from sklearn.linear_model import LinearRegression

s = 1000000 #number of rows
n = 74180464    #number of records in file

#number of training rows = 74,180,464

#skip = sorted(random.sample(range(1, n+1), n-s))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

df = pd.read_csv('../input/train.csv', header = 0, nrows = 10)#skiprows = skip)
df.columns = ['week', 'sales_depot_id', 'sales_channel_id', 'route_id', 'client_id', 'product_id', 'unit_sales_today', 'sales_today', 'unit_sales_tomorrow', 'sales_tomorrow', 'demand']
print(df)

product_dummies = pd.get_dummies(df['product_id'])
week_dummies = pd.get_dummies(df['week'])
channel_dummies = pd.get_dummies(df['sales_channel_id'])
route_dummies = pd.get_dummies(df['route_id'])
client_dummies = pd.get_dummies(df['client_id'])


df = df.join(product_dummies)
df = df.join(week_dummies)
df = df.join(channel_dummies)
df = df.join(route_dummies)
df = df.join(client_dummies)

print(df)
print(list(df.columns))


# Any results you write to the current directory are saved as output.