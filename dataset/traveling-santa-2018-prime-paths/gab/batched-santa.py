# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import csv
import sklearn.cluster as skcluster
from functools import partial
from collections import Counter
from concorde.tsp import TSPSolver
import operator
from itertools import islice

def is_prime(limit):
    limitn = limit
    not_prime = set()
    primes = [0,0]

    for i in range(2, limitn):
        if i in not_prime:
            primes.append(0)
            continue

        for f in range(i*2, limitn, i):
            not_prime.add(f)

        primes.append(1)

    return np.array(primes)
    
def make_submission(filename, path):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Path'])
        writer.writerows([p] for p in path)
    
def load_csv(filename, limit=None):
    with open(filename, newline='') as input_file:
        reader = csv.reader(input_file)
        next(reader) #Skip headers
        it = islice(reader, limit) if limit else reader
        return np.array([[float(x)*1000, float(y)*1000] for _, x, y in it])
    
def scoring_function(data):
    penalized = is_prime(len(data))
    coords = (data[:,0] + 1j * data[:,1])
    def score_path(path):
        dist = np.abs(np.diff(coords[path]))
        penalty = 0.1 * np.sum(dist[9::10] * penalized[path[9:-1:10]])
        dist = np.sum(dist)
        return dist + penalty
    return score_path

def estimate_edges(k, samples):        
    "Estimate the number of edges for k clusters of the data"
    per_cluster = samples // k
    if per_cluster == 1:
        return k**2
    return per_cluster ** 2 * k + (2*k+1)**2
    
def cluster_data(data, n_clusters, min_nodes=1):
    nodes = 0
    while nodes < min_nodes: 
        clusterer = skcluster.MiniBatchKMeans(n_clusters= n_clusters, init_size= n_clusters*3)
        labels =clusterer.fit_predict(data)
        nodes = min(Counter(labels).values())
    centroids = clusterer.cluster_centers_
    return labels, centroids
    
def concorde_solve(nodes, norm='EUC_2D', **kwargs):
        return TSPSolver.from_data(*nodes.T, norm=norm).solve(**kwargs)
        

#Test run on fewer cities, set to None to use all data
limit = None
main_time_limit = 300
clusters_time_limit = 60

#Load data
data = load_csv('../input/cities.csv', limit=limit)
    
m = len(data)
score_path = scoring_function(data)
indices = np.arange(m)

estimate_edges = partial(estimate_edges, samples = m)

#Compute k such that the number of edges between cities is minimized
n_clusters = sorted(range(1,m//2), key= estimate_edges)[0]
print(f"number of clusters: {n_clusters}, expected edges: {estimate_edges(n_clusters):,d}")

#Cluster the data into n_clusters using MiniBatchKmeans, make sure each clustr has at least 10 cities
labels, centroids = cluster_data(data[1:], n_clusters, min_nodes=1)
labels = np.insert(labels+1, 0, 0, axis=0)
centroids = np.insert(centroids, 0, data[0], axis=0)

print(len(centroids))
print('Solving TSP for main tour')
concorde_sol = concorde_solve(centroids, time_bound=main_time_limit)
assert concorde_sol.found_tour
tour = concorde_sol.tour

path = [0]

for i, cluster_idx in enumerate(tour[1:],1):
    print(f'{i} of {n_clusters}: Solving tsp for cluster {cluster_idx}')
    cluster = data[labels==cluster_idx]
    sort = sorted(range(len(cluster)), key=lambda n: (cluster[n][0], cluster[n][1]))
    cluster = cluster[sort]
    idx = indices[labels==cluster_idx][sort]
    
    solution = concorde_solve(cluster, time_bound=clusters_time_limit)
    assert solution.found_tour
    cluster_tour = solution.tour
    path.extend(idx[cluster_tour])
    
path.append(0)

# Make sure every city is visited once (except for the north pole)
assert len(set(path)) == len(data)
score = score_path(path) / 1000

make_submission('submission.csv', path)
