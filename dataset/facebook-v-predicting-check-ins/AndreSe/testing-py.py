# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df_train = pd.read_csv('../input/train.csv')

#List of possible place_ids
places = list(set(df_train['place_id'].values.tolist()))

#Get the 'first' place
place_0 = df_train.loc[(df_train['place_id']==places[0])]

#Is there a dependence between x/y-coordinates and accuracy?
print (place_0['x'].corr(place_0['accuracy']))
print (place_0['y'].corr(place_0['accuracy']))

fig = plt.figure()
fig.add_subplot(311)
sns.tsplot(place_0['accuracy'],c='r')
fig.add_subplot(312)
sns.tsplot(place_0['x'],c='b')
fig.add_subplot(313)
sns.tsplot(place_0['y'],c='g')
plt.show()

# Any results you write to the current directory are saved as output.