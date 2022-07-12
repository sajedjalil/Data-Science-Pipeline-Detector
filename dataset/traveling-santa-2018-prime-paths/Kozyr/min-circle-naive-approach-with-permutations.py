# coding: utf-8
__author__ = "https://www.kaggle.com/dkozyr"

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from heapdict import heapdict
import math
import os
from random import randint
import itertools

DINF = 1e100
N = 0 #197769
WIDTH = 5100
HEIGHT = 3400
MIN_NEIGHBOURS = 7
RADIUS_STEP = 5

INPUT_PATH = '../input/'

# ---[ prime numbers ]----------------------------------------------------------
primes = set()
def is_prime(x): return x in primes

def init_primes(max_prime):
    print('calculating prime numbers...')
    T = list(range(max_prime + 1))
    for i in range(2, max_prime + 1):
        if T[i] == i:
            for j in range(i+i, max_prime + 1, i):
                T[j] = 0
    for i in range(2, max_prime + 1):
        if T[i] == i:
            primes.add(i);

# ---[ union find ] ------------------------------------------------------------
uf = list()

def unionfind_leader(x):
    global N, uf
    if len(uf) == 0: uf = list(range(N+1))
    if uf[x] != x: uf[x] = unionfind_leader(uf[x])
    return uf[x]
    
def unionfind_is_joined(a, b):
    global uf
    return unionfind_leader(a) == unionfind_leader(b)

def unionfind_join(a, b):
    global uf
    uf[unionfind_leader(a)] = unionfind_leader(b)

# ---[ used edges ]-------------------------------------------------------------
used_edges = set()

def used_edges_insert(a, b):
    global used_edges
    if a > b: a, b = b, a
    used_edges.add((a,b))

def used_edges_has_edge(a, b):
    global used_edges
    if a > b: a, b = b, a
    return (a, b) in used_edges

def used_edges_remove(a, b):
    global used_edges
    if a > b: a, b = b, a
    used_edges.remove((a,b))

# ---[ read input data ]--------------------------------------------------------
cities = dict()

def dist(a, b):
    global cities
    A, B = cities[a], cities[b]
    dx, dy = A[0] - B[0], A[1] - B[1]
    return math.sqrt(dx*dx + dy*dy)

def read_cities():
    global N
    print('reading input data...')
    csv_cities = pd.read_csv(INPUT_PATH + 'cities.csv')
    ids = csv_cities['CityId'].values
    x = csv_cities['X'].values
    y = csv_cities['Y'].values
    for i, id in enumerate(ids):
        if x[i] < WIDTH and y[i] < HEIGHT:  # for debug purpose
            #cities[id] = (x[i], y[i])
            cities[N] = (x[i], y[i])
            N += 1

# ---[ find neighbours ]--------------------------------------------------------
neighbours = [] 

def find_neighbours():
    global neighbours
    
    print('finding neighbours...')
    neighbours = [set() for x in range(N)] 
    points = np.array(list(cities.values()))
    tree = KDTree(points)

    for n in range(N):
        idx = []
        R = 0
        while len(idx) < MIN_NEIGHBOURS:
            R += RADIUS_STEP
            idx = tree.query_ball_point(cities[n], r=R)
        neighbours[n] |= set(idx)
        for i in idx:
            neighbours[i].add(n)
        #if n % 100 == 0: print(n, neighbours[n])
    
    return

    # expand neighbours: neighbours of neighbours
    neighbours_expand = [set() for x in range(N)] 
    for i in range(N):
        neighbours_expand[i] = neighbours[i].copy()
        for x in neighbours[i]:
            neighbours_expand[i] |= neighbours[x]
        neighbours_expand[i].remove(i)
    neighbours = neighbours_expand
    
def random_neighbour(n):
    global neighbours
    w = list(neighbours[n])
    return w[randint(0, len(w) - 1)]
    
def common_neighbours(a, b):
    global neighbours
    return list(neighbours[a] & neighbours[b])
    
# ---[ evaluations ]------------------------------------------------------------
def circle_dist(path):
    d = dist(path[0], path[-1])
    for i in range(len(path) - 1):
        d += dist(path[i], path[i+1])
    return d
    
def perm_dist(a, b, path):
    d = dist(a, path[0]) + dist(b, path[-1])
    for i in range(len(path)-1):
        d += dist(path[i], path[i+1])
    return d
    
def left_path(path, idx):
    K = len(path)
    w = [path[idx]]
    i = idx - 1
    if i < 0: i = K - 1
    while i != idx:
        w.append(path[i])
        i -= 1
        if i < 0: i = K - 1
    return w
    
def right_path(path, idx):
    K = len(path)
    w = [path[idx]]
    i = (idx + 1) % K
    while i != idx:
        w.append(path[i])
        i = (i + 1) % K
    return w
    
def triangle_weight(c, e0, e1):
    return dist(c, e0) + dist(c, e1) - dist(e0, e1)
    
# ---[ path building ]------------------------------------------------------
def build_path():
    print('path building...')
    heap = heapdict()
    
    # create first triangle
    path = [randint(0, N-1)]
    path.append(random_neighbour(path[0]))
    common = common_neighbours(path[0], path[1])
    path.append(common[randint(0, len(common)-1)])
    
    unionfind_join(path[0], path[1])
    unionfind_join(path[1], path[2])

    used_edges_insert(path[0], path[1]);
    used_edges_insert(path[1], path[2]);
    used_edges_insert(path[2], path[0]);
    
    P = len(path)
    for i in range(P):
        e0, e1 = path[i], path[(i+1)%P]
        common = common_neighbours(e0, e1)
        for c in common:
            if not unionfind_is_joined(c, e0):
                heap[(c, e0, e1)] = triangle_weight(c, e0, e1)

    while len(path) < N:
        P = len(path)
        triangle = heap.popitem()[0]
        if used_edges_has_edge(triangle[1], triangle[2]) and not unionfind_is_joined(triangle[0], triangle[1]):
            used_edges_remove(triangle[1], triangle[2])
            used_edges_insert(triangle[0], triangle[1])
            used_edges_insert(triangle[0], triangle[2])
            
            unionfind_join(triangle[0], triangle[1])
            unionfind_join(triangle[0], triangle[2])
            
            if path[0] in [triangle[1], triangle[2]] and path[-1] in [triangle[1], triangle[2]]:
                path.insert(0, triangle[0])
            else:
                for i in range(P):
                    if path[i] in [triangle[1], triangle[2]]:
                        path.insert(i + 1, triangle[0])
                        break

            # edge a--b
            common = common_neighbours(triangle[0], triangle[1])
            for c in common:
                if not unionfind_is_joined(c, triangle[1]):
                    heap[(c, triangle[0], triangle[1])] = triangle_weight(c, triangle[0], triangle[1])
            
            # edge a--c
            common = common_neighbours(triangle[0], triangle[2])
            for c in common:
                if not unionfind_is_joined(c, triangle[2]):
                    heap[(c, triangle[0], triangle[2])] = triangle_weight(c, triangle[0], triangle[2])
                    
            if P % 1000 == 0: print('path size: ', P, ', circle distance: ', circle_dist(path))
            
        if len(heap) == 0 and len(path) != N:
            for i in range(N):
                if not unionfind_is_joined(i, path[0]):
                    for x in range(len(path)-1):
                        heap[(i, path[x], path[x+1])] = triangle_weight(i, path[x], path[x+1])

    return path

# ---[ path permutations ]------------------------------------------------------
def path_permutation(path, K):
    print('path permutation...')
    for i in range(len(path) - K):
        a, b = path[i], path[i + K - 1]
        w = path[(i+1):(i+K-1)]
        w.sort()
        
        best, D = [], DINF
        for x in itertools.permutations(w):
            d = perm_dist(a, b, x)
            if D > d:
                D, best = d, x
        
        for x in range(len(best)):
            path[i + 1 + x] = best[x]

    return path

# ---[ select best path ]-------------------------------------------------------
def select_best_path(path):
    for i in range(len(path)):
        if path[i] == 0:
            path = path[i:] + path[:i]
            break
    return path + [0]

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    init_primes(200000)
    
    read_cities()
    #print(N, evaluation(list(range(N))))
    
    find_neighbours()

    for iter in range(1):
        path = build_path()
        print('path size: ', len(path), ', circle distance: ', circle_dist(path))

    permutations = {7:3, 8:2}
    for perm in permutations:
        for iter in range(permutations[perm]):
            path = path_permutation(path, perm)
            print('path size: ', len(path), ', circle distance: ', circle_dist(path))
    
    path = select_best_path(path);

    # save submission
    with open('submission.csv', 'w') as submit_file:
        submit_file.write('Path\n')
        for x in path:
            submit_file.write('%d\n' % x)
            
    print('done')