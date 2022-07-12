# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
#print (train.head())

#test = pd.read_csv("../input/test.csv",index_col='Id')
#print (test.head())
print (train.dtypes)

train['Dates'] = train['Dates'].astype('datetime64')
#train['Dates'] = train['Dates'].astype('str')
#print (train.dtypes)
#train['Dates'] = pd.to_datetime(train['Dates'], format='%d.%m.%Y %H:%M')
#train['Dates'] = train.to_datetime(str(train['Dates']) )
train['Category'] = train['Category'].astype('category')
train['Descript'] = train['Descript'].astype('category')
train['DayOfWeek'] = train['DayOfWeek'].astype('category')
train['PdDistrict'] = train['PdDistrict'].astype('category')
train['Resolution'] = train['Resolution'].astype('category')
train['Address'] = train['Address'].astype('category')

print (train.dtypes)
print (train.head())
#print (str(train['Dates']) + ' ' + str(train['Dates'].year) + ' ' + str(train['Dates'].time))