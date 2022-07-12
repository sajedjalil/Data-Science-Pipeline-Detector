# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualisation
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.

df= pd.read_csv('../input/application_train.csv', header=0 ,sep = ',')

#what is the distribution of no of children for the loan applicants 
sns.set_style('white')
sns.boxplot(data=df, x='TARGET',y= 'CNT_CHILDREN')
sns.despine()

df1=df.groupby(['TARGET',df.columns[13]]).size()    
df1 =df1.reset_index()  
df1.rename(columns={0: 'count'}, inplace= True) 
# education Vs Target
ax = sns.barplot(x=df1.columns[1], y="count", hue="TARGET", data=df1)
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

#income Vs Target
df2=df.groupby(['TARGET',df.columns[12]]).size()    
df2 =df2.reset_index()  
df2.rename(columns={0: 'count'}, inplace= True) 

ax = sns.barplot(x=df2.columns[1], y="count", hue="TARGET", data=df2)
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
