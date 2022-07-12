
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas
from sklearn.cluster import KMeans





def clusteriser(matrice,listindice,par):
    clf=KMeans(n_clusters=par)
    a=clf.fit_predict(matrice[:,:2])
    coordcenter=clf.cluster_centers_
    one=list(set(a))
    one.sort()
    occ=[]
    for i in one:
        occ.append(one.count(i))
    listematrice=[np.zeros(shape=(1,3))]*len(one)
    listeindex=[]
    for i in range(len(one)):
        listeindex.append([])
    for i in range(np.size(matrice,0)):
        listematrice[a[i]]=np.concatenate((listematrice[a[i]],[matrice[i]]),axis=0)
        listeindex[a[i]].append(listindice[i])
    return listematrice,listeindex,coordcenter