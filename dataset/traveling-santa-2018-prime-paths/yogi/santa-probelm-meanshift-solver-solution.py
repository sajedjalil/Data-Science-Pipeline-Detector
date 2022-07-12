import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sympy import isprime, primerange
#from altair import *
#import seaborn as sns; sns.set()
import math
from sklearn.model_selection import KFold
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
print(os.listdir("../input"))
from sklearn.cluster import DBSCAN
# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', 500)
from scipy import spatial
from scipy.spatial import distance_matrix
from scipy.spatial import distance
cities = pd.read_csv("../input/cities.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")
from tsp_solver.greedy import solve_tsp
from scipy.spatial.distance import cdist,pdist
from sklearn.cluster import Birch
from collections import deque
from sklearn.neighbors import KDTree
from matplotlib import collections  as mc
import pylab as pl
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle
import random

random.seed(758)
plot_graphs = False

def plot_tour2(df,thepath):
    lines = [[(df.X[thepath[i]],df.Y[thepath[i]]),(df.X[thepath[i+1]],df.Y[thepath[i+1]])] for i in range(0,len(df)-1)]
    lc = mc.LineCollection(lines, linewidths=2)
    fig, ax = pl.subplots(figsize=(30,30))
    ax.set_aspect('equal')
    ax.add_collection(lc)
    ax.autoscale()
    
def check_plot_tour(df,path_city,tot_len):
    if tot_len == 0:
        track_counter = np.arange(tot_len+1,len(df)+ tot_len+1).tolist()
    else:
        track_counter = np.arange(tot_len-1,len(df)+ tot_len).tolist()
    df = df.loc[path_city]
    df['track_counter_10'] = [1 if v == 0 else 0 for v in [t % 10 for t in track_counter]]
    df['match'] = (df['track_counter_10'] * df['isPrime']*2) + df['isPrime']
    df.loc[path_city[0],'match'] = 6
    df.loc[path_city[len(path_city)-1],'match'] = 5
    if plot_graphs:
        colo = {'1':'blue', '0':'black', '2':'green', '3':'red','4':'orange', '5':'pink', '6':'yellow'}
        lines = [[(df.X[path_city[i]],df.Y[path_city[i]]),(df.X[path_city[i+1]],df.Y[path_city[i+1]])] for i in range(0,len(df)-1)]
        lc = mc.LineCollection(lines, linewidths=.5)
        fig, ax = pl.subplots(figsize=(30,30))
        ax.set_aspect('equal')
        ax.add_collection(lc)
        ax.scatter(df.X, df.Y, c=df['match'].astype(str).apply(lambda x: colo[x]),s=(df['match']+1)*100)
        for i, txt in enumerate(path_city):
            ax.annotate(df.CityId[path_city[i]], (df.X[path_city[i]], df.Y[path_city[i]]))
        ax.autoscale()
        plt.show()

def calc_distance(traversing_cities):
    print("Calculating traversing distance .... ")
    dist = []
    for ddd in np.arange(len(traversing_cities))[1:]:
        t = cdist(traversing_cities.iloc[[ddd,ddd-1]][['X','Y']],traversing_cities.iloc[[ddd,ddd-1]][['X','Y']],'euclidean').tolist()[0][1]
        dist.append(t)
        #print(ddd)
    dist.insert(0,cdist(traversing_cities.iloc[[ddd,0]][['X','Y']],traversing_cities.iloc[[ddd,0]][['X','Y']],'euclidean').tolist()[0][1])
    traversing_cities['dist'] = dist
    print("Calculating penalities")### 1702332.52
    reaching_back = cdist(traversing_cities.iloc[[len(traversing_cities)-1,0]][['X','Y']],traversing_cities.iloc[[len(traversing_cities)-1,0]][['X','Y']],'euclidean').tolist()[0][1]
    rep_list = [0,0,0,0,0,0,0,0,0,1]
    traversing_cities['idx_10'] = (rep_list * int(len(traversing_cities)/10)) + rep_list[:len(traversing_cities) % 10]  # pd.Series(traversing_cities.index).apply(lambda x: 1 if x % 10 == 0 else 0 ).astype(int)
    traversing_cities['penality'] = traversing_cities[['idx_10','isPrime']].apply(lambda x: x[0] & x[1], axis = 1)
    
    traversing_cities['total_distance'] = traversing_cities[['idx_10','penality','dist']].apply(lambda x: x[2]*1.1  if (x[0] == 1 and x[1] == 0)  else x[2],axis = 1)
    tot_dist = str(traversing_cities['total_distance'].sum() + reaching_back)
    print("Total Distance  : " + tot_dist)
    return tot_dist


def check_distance(XS,track_counter,path_city):
    df = XS.loc[path_city]
    t = cdist(df[['X','Y']].values,df[['X','Y']].values)
    t = np.triu(t)
    t = t[:,1:].diagonal().tolist()
    t.insert(0,0)
    df['dist'] = t
    #print("Calculating penalities")### 1702332.52
    df['idx_10'] = [1 if v == 0 else 0 for v in [t % 10 for t in track_counter]]
    df['penality'] = df[['idx_10','isPrime']].apply(lambda x: ((x[0] == 1) & (x[1] == 0)).astype(int), axis = 1)
    df['total_distance'] = df[['idx_10','penality','dist']].apply(lambda x: x[2]*1.1  if (x[1] == 1)  else x[2],axis = 1)
    df['match'] = ((df['idx_10'] == df['isPrime']) & (df['idx_10'] > 0)).astype(int) #* df['idx_10']
    tot_dist = str(df['total_distance'].sum())
    no_hit = df.match.sum() #print("Total Distance  : " + tot_dist)
    total_prime = df.isPrime.sum()
    return df,tot_dist,no_hit,total_prime

    
cities['isPrime'] = cities.CityId.apply(isprime).astype(int)
cities = cities.sort_values('CityId', ascending = True).reset_index(drop=True)
cities['pts'] = 1
cities['pts'] = cities['pts'] - cities['isPrime']

laby = 150
labx = int(1.5*laby)

import itertools
brk = 70
lst = range(0,brk)
my = list(itertools.chain.from_iterable(itertools.repeat(x, int(197800/brk)+10) for x in lst))
cities = cities.sort_values('X', ascending = True).reset_index(drop=True)
cities['catX'] = my[:len(cities)]#pd.cut(cities['X'],labx, labels=np.arange(0,labx,1)).astype(int)
cities = cities.sort_values('Y', ascending = True).reset_index(drop=True)
cities['catY'] = my[:len(cities)] #pd.cut(cities['Y'],laby, labels=np.arange(0,laby,1)[::-1]).astype(int)
cities['tot'] = 1
cities['grp_idx'] = cities.catX.astype(str) + "-" + cities.catY.astype(str)


#tab = pd.pivot_table(cities, values='tot', index=['catY'], columns=['catX'], aggfunc=np.sum)
#tab = np.array(tab.sort_index(ascending = False).fillna(0))
#plt.imshow(tab)
#plt.show()

## Cluster Approach
def plot_mean_clust(Xp,n_clusters_,cluster_centers):
 plt.figure(1)
 plt.clf()
 colors_2 = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

 for k, col in zip(range(n_clusters_), colors_2):
     my_members = labels == k
     cluster_center = cluster_centers[k]
     plt.plot(Xp[my_members, 0], Xp[my_members, 1], col + '.')
     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
              markeredgecolor='k', markersize=1)
 plt.title('Estimated number of clusters: %d' % n_clusters_)
 plt.show()
    
    
Xp = cities[cities.isPrime == 1][['X','Y']].as_matrix()
Xnp = cities[cities.isPrime == 0][['X','Y']].as_matrix()
X = np.append(Xp,Xnp,axis = 0)
bandwidth = estimate_bandwidth(X, quantile=.001)
ms = MeanShift(bandwidth=bandwidth, bin_seeding= True ,min_bin_freq = 10,cluster_all = True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)                 
cities['grp_idx'] = ms.predict(cities[['X','Y']]).tolist()
print("Plot for clusters generated")
plot_mean_clust(X,n_clusters_,cluster_centers) 

#######

#grp_max = cities.groupby(['grp_idx'])['X','Y'].max().reset_index()
#grp_max.head(5)

#D = distance_matrix(np.array(grp_max[['X','Y']]),np.array(grp_max[['X','Y']]))
#D = np.tril(D)
#strt_grp_idx = cities.loc[cities.CityId == 0]['grp_idx'].tolist()[0]
#strt_idx = np.where(grp_max.grp_idx == strt_grp_idx)[0].tolist()[0]
#end_options = D[strt_idx]
#end_options = end_options.tolist()
#end_idx = end_options.index(min(i for i in end_options if i > 0))
##end_grp_idx = grp_max.loc[end_options_idx]['grp_idx']
#print("Calculating  group optimal path")
#path = solve_tsp(D, endpoints = (strt_idx,end_idx))
#traverse_grp = pd.DataFrame(np.array([[i,grp_max.loc[v,'grp_idx']] for i,v in enumerate(path)]))
#traverse_grp.columns = ['priority','grp_idx']
#
#cities = cities.merge(traverse_grp, left_on=['grp_idx'], right_on=['grp_idx'], how='left')
#cities = cities.sort_values('CityId').reset_index()
#print("Creating traversing index from X")
#priority_labels = traverse_grp.priority.tolist()
#cities.to_csv("cities_mod.csv",index = False)

#cities = pd.read_csv("cities_mod.csv")
#del(cities['level_0'])
#del(cities['index'])

#priority_labels = sorted(cities.priority.unique().tolist())
#traversing_grp_idx_in_X = [cities[cities.priority == pl].index.tolist() for pl in priority_labels]
    
X = cities[['CityId','X','Y','isPrime','grp_idx']].reset_index()  # Start 
cities = cities.sort_values('CityId').reset_index()
the_traversing = list()
grab_last = 0
the_distance = 0
tot_hits = 0
sum_dist = 0.0
grp_cnt = 0
strt_grp = cities.loc[cities.CityId == 0,'grp_idx'].tolist()[0]
XS = cities[cities['grp_idx'] == strt_grp][['CityId','X','Y','isPrime']]
while grp_cnt <= n_clusters_:
    
    DS = distance_matrix(np.array(XS.iloc[:,[1,2]]),np.array(XS.iloc[:,[1,2]]))  # calculate distance within cities
    #DS = np.tril(DS)  # kept only lower triangle 
    strt_idx = 0 #np.argmin(np.min(DS, axis=1))  # get min distance 
    idx_opts = DS[:,1:].diagonal().tolist()
    possible_end_idx = sorted(range(len(idx_opts)), key=lambda i: idx_opts[i])  # max at RHS 
    sel_end_idx = [i for i in possible_end_idx if i not in [strt_idx]][-30:][1::10]
    
    tot_len = len(the_traversing)
    track_count = np.arange(tot_len+1,tot_len+len(XS)+1).tolist()
        
    for end_rel_idx,end_idx in enumerate(sel_end_idx):
        print(end_rel_idx)
        path_s = solve_tsp(DS, endpoints = (strt_idx,end_idx))
        path_city = XS.iloc[path_s].index.tolist()
        df,curr_dist,no_hit,total_prime = check_distance(XS.iloc[path_s],track_count,path_city)
        if end_rel_idx == 0:
            check_dist = curr_dist
            check_hit = no_hit
            sel_path = path_city
        else:
            if (curr_dist < check_dist): #& (no_hit > check_hit):
                sel_path = path_city
#                    plot_graphs = False
#                    check_plot_tour(cities.iloc[sel_path],sel_path,tot_len)
#                    plot_graphs = False
                check_hit = no_hit
                check_dist = curr_dist
                print("hit it")
                print(str(check_dist))
    if grp_cnt < n_clusters_:  
        print("Selection")
        ## remove grp from dataframe 
        X = X[X['grp_idx'] != strt_grp]
        tree = KDTree(X[['X','Y']].values, leaf_size=10)
        # get index of nearest point  
        last = XS.iloc[path_s][-1:][['X','Y']].values.tolist()
        dist, ind = tree.query(last, k=1)
        # find the grp 
        ind = ind.tolist()[0][0]
        strt_grp = int(X.iloc[ind]['grp_idx'])
        # make nearest point as start for that group
        X2 = X[X['grp_idx'] == strt_grp][['CityId','X','Y','isPrime']]
        DSN = distance_matrix(np.array(XS.iloc[:,[1,2]]),np.array(X2.iloc[:,[1,2]]))  # calculate distance within cities
        idx_opts = np.min(DSN, axis=0).tolist()
        grab_last = X2.iloc[np.argmin(idx_opts)]['CityId']
        X2.index = X2.CityId.tolist()
        index_created = X2.index.tolist()
        index_created.remove(grab_last)
        index_created.insert(0,int(grab_last))
        XS = X2.loc[index_created]
    # findX end recusiverly  
    #XS = X2[X2.CityId ! = grab_last]
    sel_city = sel_path
    tot_hits = tot_hits + check_hit
    sum_dist = sum_dist + float(check_dist)
    the_traversing.extend(sel_city)
    grp_cnt = grp_cnt + 1
    print("  Total Visits : " + str(len(the_traversing)) + "  dist/hits/total: "+ str(sum_dist)  + " /" + str(check_hit) + " /" + str(total_prime) + " total hits :: " + str(tot_hits))


sample_submission.to_csv("submission_2018_01_08.csv",index = False)

traversing_cities = cities.iloc[the_traversing[:-1]].copy()
plot_tour2(traversing_cities,the_traversing[:-1])