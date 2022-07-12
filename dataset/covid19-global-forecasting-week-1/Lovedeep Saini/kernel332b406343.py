# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns  
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


df_train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")
print(df_train.head())

print(df_train.info())

for i in df_train.columns:
    print(i, df_train[i].nunique())


tmp = df_train.groupby(["Country/Region"])["ConfirmedCases","Fatalities"].max().reset_index(level="Country/Region")
tmp20 = tmp.sort_values(by=["ConfirmedCases"], ascending=False).head(20)
tmp20.head()
plt.figure(figsize=(27,5))
sns.barplot(x='Country/Region',
            y='ConfirmedCases',
            data=tmp20)

plt.figure(figsize=(27,5))
sns.barplot(x='Country/Region',
            y='Fatalities',
            data=tmp20)



df_train["Date"] = pd.to_datetime(df_train["Date"])


df_subset = df_train[['Date','Country/Region','Fatalities']]
df2 = df_subset.groupby(["Date"])['Fatalities'].max().reset_index()
df3 = df_subset.groupby(["Country/Region"])["Fatalities"].max().reset_index()
df3 = df3[df3["Fatalities"]>200].sort_values(by=['Fatalities'],ascending=False)

#Keep only those countries which have more than 100 fatalties so far.
df_majors = pd.merge(df_train,df3['Country/Region'],on='Country/Region',how='inner')
df_majors.head(10)
df3 = df_majors.groupby(["Date","Country/Region"])['Fatalities'].sum().reset_index()

import datetime as Dr
countries = df3['Country/Region'].unique().tolist()
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5),sharey=True)

ax1.plot(df2["Date"],df2['Fatalities'],marker='o',linewidth=2,label='Global')
xticks = [Dr.datetime.strftime(t,'%Y-%m-%d') for t in df2["Date"].to_list()]
xticks = [tick for i,tick in enumerate(xticks) if i%4==0 ]

for country in countries:
    df4 = df3[df3['Country/Region']==country]
    ax2.plot(df4["Date"],df4['Fatalities'],marker='o',linewidth=2,label=country)

for ax in (ax1,ax2):
    ax.legend()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks,rotation=90)
    ax.set_xlabel('Date')
    ax1.set_ylabel('Fatalities')
    ax.yaxis.grid(linestyle='dotted')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
ax1.set_title('Time Evolution of Global Fatalities')
ax2.set_title('Time Evolution of National Fatalities')






    