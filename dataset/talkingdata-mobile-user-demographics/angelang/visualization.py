# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
#matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
import matplotlib.cm as cm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
datadir = '../input'

# Any results you write to the current directory are saved as output.
#two columns that show which train or test set row a particular device_id belongs to.

# Any results you write to the current directory are saved as output.

#train.group.value_counts().sort_index(ascending=False).plot(kind = 'bar')
#plt.title('Age and Genger Group Distribution')
#plt.ylabel('Count')
#plt.xlabel('Group Name')
#plt.savefig('age_gender.png')

#train.gender.value_counts().plot(kind='barh')
#plt.title('Gender Distribution')
#plt.ylabel('Count')
#plt.xlabel('Gender')
#plt.savefig('gender.png')

#plt.title('Age Distribution')
#sns.distplot(train.age,ax=ax)
#plt.savefig('age.png')
phone = pd.read_csv('../input/phone_brand_device_model.csv')
phone = phone.drop_duplicates('device_id', keep='first')
phone_size = phone.groupby('device_id').size()
phone_idx = phone_size[phone_size > 1]
len(phone_idx) ## duplicate
phone.loc[phone.device_id.isin(phone_idx.index),:].head()
count = phone.groupby(['device_model'])['phone_brand'].apply(pd.Series.nunique)
count.value_counts()
lebrand = LabelEncoder().fit(phone.phone_brand.values)
phone['brand'] = lebrand.transform(phone.phone_brand.values)
m = phone.phone_brand.str.cat(phone.device_model)
lemodel = LabelEncoder().fit(m)
phone['model'] = lemodel.transform(m)
#phone['model'].plot(kind='hist',bins=50)
#plt.title('Model Distribution')
#plt.xlabel('Model')

#phone['brand'].plot(kind='hist',bins=50)
#plt.title('Brand Distribution')
#plt.xlabel('Brand')

#events.shape[0] - events['is_installed'].value_counts()
#events['event_id'].value_counts().plot(kind='hist',bins=50)
#plt.title('Event Distribution')
#plt.ylabel('Count')
#plt.xlabel('Event')
#plt.savefig('event.png')
train = pd.read_csv('../input/gender_age_train.csv')
#print(devicelabels[['device_id','label']])
#print(train)
metrain = train.merge(phone[['device_id','brand','model']], how='left',on='device_id')
print(metrain)
def plot_by(df, cat, by, perc = 1):
    # Find popular categories
    c = df[by].value_counts().cumsum()/df.shape[0]
    take = c[c<=perc].index
    # Pool rare categories into 'other' cat
    gr = df[by].copy()
    gr[~(df[by].isin(take))] = 'other'
    # Count target classes in groups
    c = df.groupby([gr,cat]).size().unstack().fillna(0)
    total = c.sum(axis=1)
    meanprobs = c.sum(axis=0).cumsum()/df.shape[0]
    # Transform to probabilities
    sortcol = c.columns[int(np.floor((c.shape[1]-1)/2))]
    c = c.div(c.sum(axis=1), axis='index')
    # Cumsum for stacked bars
    c = c.cumsum(axis=1).sort_values(by=sortcol)
    total = total.loc[c.index]
    # Prepare plot data
    left = np.array([0, *(total.iloc[:-1].cumsum().values)])
    ticks = left + 0.5*total.values
    colors = cm.rainbow(np.linspace(0.1,0.9,num=c.shape[1]))
    fig, ax = plt.subplots(figsize=(10,8))
    for (i,col) in enumerate(c.columns[::-1]):
        height = c[col].values
        ax.bar(left, height, total.values,label=col, color = colors[i],zorder = c.shape[0]+i)
    ax.legend(bbox_to_anchor=(1.1, 1),title=cat,fontsize = 4.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(total.index, rotation='vertical')
    ax.set_xlim(0,left[-1]+total.values[-1])
    return ax
    
ax = plot_by(metrain, 'age','model')
plt.ylabel('probabilities')
plt.xlabel('model')
plt.title('Age distribution based on phone models')
plt.savefig('age_on_model.png')
