import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

mpl.style.use('ggplot')
train=pd.read_csv('..//input//train.csv')
train=train.drop(['ID','target'],axis=1)

df=pd.isnull(train).astype(int) #Get NA as 1, everything else as 0
df=df[df.sum(axis=0).sort_values(axis=0, ascending=False).index] #Sort columns by number of NA

#distribution of NA by column
fig, ax = plt.subplots(figsize=(10, 5))
ax.grid(False)
bar = ax.bar(range(len(df.columns)),df.sum(axis=0).values)
plt.xticks(0.5+np.arange(len(df.columns)), df.columns, fontsize=8,rotation=90)
plt.axis('tight')
plt.tight_layout()
plt.savefig('BNP_NA_BAR.png')

t=np.zeros((train.shape[0]), dtype=object) #130 bit int will be stored here for sorting purposes

for i,c in enumerate(train.columns):
    t+=df[c]*(2**i)#Create 130bit long integer with 1 corresponding to NA

df=pd.concat((df,pd.DataFrame(t)),axis=1)#add as a row
df=df.sort_values(by=0, ascending=False).drop(0,axis=1)#and sort by it

fig, ax = plt.subplots(figsize=(10, 10))

ac=KMeans(n_clusters=5)#Use clusters for coloring
clusters=(ac.fit_predict(df)+1).reshape(-1,1)

ax.imshow(df.values*clusters, aspect=0.001, cmap=plt.cm.viridis, interpolation='none')
plt.xticks(0.5+np.arange(len(df.columns)), df.columns, fontsize=8,rotation=90)
ax.grid(False)

ax.spines['left'].set_color('steelblue')
ax.spines['right'].set_color('steelblue')
ax.spines['top'].set_color('steelblue')
ax.spines['bottom'].set_color('steelblue')

ax.xaxis.grid(True, which='major', color='steelblue',alpha=0.3)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.tight_layout()
plt.savefig('BNP_NA_2.png')
