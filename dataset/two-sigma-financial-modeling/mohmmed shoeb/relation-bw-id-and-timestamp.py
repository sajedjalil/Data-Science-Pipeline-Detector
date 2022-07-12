import kagglegym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

p = sns.color_palette()



# load data
hdf = pd.HDFStore("../input/train.h5")
df = pd.concat([hdf.select(key) for key in hdf.keys()])
hdf.close()




print('Unique Ids: ', df['id'].nunique())




# id counts w.r.t time
temp = df.groupby('timestamp').apply(lambda x: x['id'].nunique())
plt.figure(figsize=(8,4))
plt.plot(temp, color=p[0])
plt.xlabel('timestamp')
plt.ylabel('id count')
plt.title('Number of ids over time')


# lifespan of each id
temp = df.groupby('id').apply(len)
temp = temp.sort_values()
temp = temp.reset_index()
plt.figure(figsize=(8,4))
plt.plot(temp[0], color=p[0])
plt.xlabel('index for each id sorted by number of timestamps')
plt.ylabel('number of timestamps')
plt.title('Number of timestamps ("Lifespan") for each id')
print(temp[0].describe())




N= 100
temp2 = df[df['id'].isin(temp['id'].head(N).values)]
temp2 = temp2.sort_values(['id', 'timestamp'])
temp2 = temp2.pivot(index='timestamp', columns='id', values='id')
plt.figure(figsize=(8,4))
plt.plot(temp2)
plt.xlabel('timestamp')
plt.ylabel('id')
plt.title('"Lifespan" for the {} shortest lived ids'.format(N))



n_start = 700
n_end = 750
temp2 = df[df['id'].isin(temp['id'][n_start:n_end].values)]
temp2 = temp2.sort_values(['id', 'timestamp'])
temp2 = temp2.pivot(index='timestamp', columns='id', values='id')
plt.figure(figsize=(8,4))
plt.plot(temp2)
plt.xlabel('timestamp')
plt.ylabel('id')
plt.title('"Lifespan" for ids ranked from {}-{}'.format(n_start, n_end))



print('Ids with timestamp=0: ', len(df[df['timestamp'] == 0]))
print('Ids with timestamp=max: ', len(df[df['timestamp'] == df['timestamp'].max()]))
print('Total ids: ', df['id'].nunique())
