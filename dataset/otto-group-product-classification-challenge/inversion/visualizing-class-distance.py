import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10., 10.)
sns.set()

from numpy.random import choice
from scipy.spatial.distance import pdist, squareform

os.system("ls ../input")


train = pd.read_csv("../input/train.csv", index_col='id')
label = train.target.apply(lambda x: int(x[-1])-1).values
train.drop('target', axis=1, inplace=True)

#
# Pick random samples from each class
#

samples_per_class = 1000
sample_idx = []

for c in range(9):
  choices = choice(np.where(label==c)[0], samples_per_class, replace=False)
  sample_idx.extend(choices)

samples = train.loc[sample_idx]

#
# create a distance matrix for the samples
#

dm = squareform(pdist(samples.values, 'correlation'))
dm = dm / dm.max()

#
# Distance of each point from the edges and middle of each class (distribution)
#

percentiles = [10,50,90]
cols = [str(c)+'_'+str(p) for c in range(9) for p in percentiles]
dist_metric = pd.DataFrame(index=range(samples_per_class*9), columns=cols)

# I <3 Loops

for i in range(samples_per_class*9):
  for c in range(9):
    start = c*samples_per_class
    class_dists = dm[i,start:start+samples_per_class]
    for p in percentiles:
      col = str(c)+'_'+str(p)
      dist_metric.loc[i,col] = np.percentile(class_dists,p)  # << This is the important line

# Now each point has a set of features for how far it is from the different classes
# You can inspect the data cross-plotted atainst different features.
# For example, this shows the the distance each point is from the 10th
# percentile of the points in Class 4 against the 10th percentile of 
# the points in Class 6

plt.rcParams['figure.figsize'] = (6., 6.)
plt.scatter(dist_metric.loc[1000:2000,'3_10'], dist_metric.loc[1000:2000,'5_10'], color='green', alpha=0.5, label='Class 2')
plt.scatter(dist_metric.loc[2000:3000,'3_10'], dist_metric.loc[2000:3000,'5_10'], color='blue', alpha=0.5, label='Class 3')
plt.scatter(dist_metric.loc[5000:6000,'3_10'], dist_metric.loc[5000:6000,'5_10'], color='red', alpha=0.5, label='Class 5')
plt.title('Correlation Distance')
plt.legend(loc='upper right')
plt.xlim((0,1))
plt.ylim((0,1))
plt.savefig('distance_metric.png')


# If you want to look at all the cross plots

#for i in dist_metric.columns:
#  for j in  dist_metric.columns:
#    plt.rcParams['figure.figsize'] = (6., 6.)
#    plt.scatter(dist_metric.loc[1000:2000,i], dist_metric.loc[1000:2000,j], color='green', alpha=0.5)
#    plt.scatter(dist_metric.loc[2000:3000,i], dist_metric.loc[2000:3000,j], color='blue', alpha=0.5)
#    plt.scatter(dist_metric.loc[5000:6000,i], dist_metric.loc[5000:6000,j], color='red', alpha=0.5)
#    plt.title('Correlation - {} - {}'.format(i,j))
#    plt.xlim((0,1))
#    plt.ylim((0,1))    
#    plt.show()

