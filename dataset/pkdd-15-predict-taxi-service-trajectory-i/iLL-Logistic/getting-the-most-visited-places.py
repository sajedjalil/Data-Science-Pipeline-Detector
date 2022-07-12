import datetime, gc, random
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt, zipfile

# reading training data
zf = zipfile.ZipFile('../input/train.csv.zip')
df = pd.read_csv(zf.open('train.csv'), converters={'POLYLINE': lambda x: json.loads(x)[-1:]})
latlong = np.array([[p[0][1], p[0][0]] for p in df['POLYLINE'] if len(p)>0])

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

range_n_clusters = [8]

for n_clusters in range_n_clusters:    
    clusterer = KMeans(n_clusters=n_clusters, verbose =1, n_jobs = -1).fit(latlong)
    cluster_labels = clusterer.labels_
    silhouette_avg = silhouette_score(latlong, cluster_labels, sample_size = 10000)
    print("For n_clusters = ", n_clusters," The average silhouette_score is : ", silhouette_avg)

