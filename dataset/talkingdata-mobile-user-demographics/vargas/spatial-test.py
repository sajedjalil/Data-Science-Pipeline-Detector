# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import DBSCAN

df = pd.read_csv('../input/events.csv',nrows=100000).dropna()
coordinates = df.as_matrix(columns=['longitude', 'latitude'])
db = DBSCAN(eps=.01, min_samples=10, algorithm='ball_tree')
db.fit(coordinates)

cluster_labels = db.labels_
num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print('Number of clusters: {}'.format(num_clusters))

clusters = pd.Series([coordinates[cluster_labels == n] for n in range(num_clusters)])
