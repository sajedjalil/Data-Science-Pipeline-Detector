# %% [code]
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import comb
import math

#%% Load dataset
train = pd.read_csv( '../input/osic-pulmonary-fibrosis-progression/train.csv' )
#%% ######################## Table level
## Number of rows and columns
train.shape
#%% Name of columns
list(train)
#%% Checking for missing values
print(train.isna().sum())
#%%
"""

                        Univariate analysis
                    
                    
"""
#%% Variable names
list(train)
#%% Datatype of each variable
train.dtypes
#%% number of unique values in each object variable
ObjectVariables = train.dtypes[train.dtypes==object].index
FrequencyOfUniqueValues=[]
for i in ObjectVariables:
    FrequencyOfUniqueValues.append(len(train[i].unique()))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(ObjectVariables,FrequencyOfUniqueValues)
plt.show()
#%% Frequency distribution of object variables
uniqueValue=[]
for i in ObjectVariables:
    uniqueValue = train[i].unique()
    Frequency=[]
    for j in uniqueValue:
        Frequency.append(len(train[train[i]==j]))
    mydf = pd.DataFrame()
    mydf['uniqueValue'] = uniqueValue
    mydf['Frequency'] = Frequency
    mydf.sort_values(by='Frequency', inplace=True)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(mydf.uniqueValue,mydf.Frequency)
    ax.legend(labels=[i, j])
    plt.show()
#%% Frequency distribution of non-object variables (For density distribution drop kde and norm_hist)
Nonobjecttypes = train.dtypes[train.dtypes!=object].index
f, Axes = plt.subplots(len(Nonobjecttypes),1 , figsize = (10, 20))

for i in range(len(Nonobjecttypes)):
    sns.distplot(train[Nonobjecttypes[i]],kde=False,bins = 100, norm_hist=False, ax=Axes[i], hist=True)
#%%
"""

                        Bivariate analysis
                    
                    
"""
#%%
# Compute Correlation
fac = math.factorial
n = len(Nonobjecttypes)
comb = int(fac(n)/2/fac(n-2))

f, Axes = plt.subplots(comb,1 , figsize = (10, 20))
scatterplot=-1
for i in range(len(Nonobjecttypes)-1):
    for j in range(i+1, len(Nonobjecttypes)):
        scatterplot=scatterplot+1

        cor, _ = pearsonr(train[Nonobjecttypes[i]], train[Nonobjecttypes[j]])
        print (Nonobjecttypes[i] + ":"+Nonobjecttypes[j] +" = "+ str(cor))
        ax.set_title("Graph (a)")
        sns.scatterplot(hue = train['Sex'],x = train[Nonobjecttypes[i]], y = train[Nonobjecttypes[j]], ax=Axes[scatterplot])

#%%


#%%
"""

                        Multivariate analysis
                    
                    
"""
#%% Correlation with other object variables

fac = math.factorial
n = len(Nonobjecttypes)
comb = int(fac(n)/2/fac(n-2))

f, Axes = plt.subplots(comb*(len(ObjectVariables)-1),1 , figsize = (10, 60))
scatterplot=-1
for h in ObjectVariables[1:]:#Excluding patient
    for i in range(len(Nonobjecttypes)-1):
        for j in range(i+1, len(Nonobjecttypes)):
            scatterplot=scatterplot+1    
            cor, _ = pearsonr(train[Nonobjecttypes[i]], train[Nonobjecttypes[j]])
            print (Nonobjecttypes[i] + ":"+Nonobjecttypes[j] +" = "+ str(cor))
            ax.set_title("Graph (a)")
            sns.scatterplot(hue = train[h],x = train[Nonobjecttypes[i]], y = train[Nonobjecttypes[j]], ax=Axes[scatterplot])

#%% change in FVC across the weeks in Patients

plt.figure(figsize = (20, 6))
a = sns.lineplot(x = train["Weeks"], y = train["FVC"], hue = train["Patient"], legend=False, size=1)

#%%
plt.figure(figsize = (20, 6))
a = sns.lineplot(x = train["Weeks"], y = train["Percent"], hue = train["Patient"], legend=False, size=1)