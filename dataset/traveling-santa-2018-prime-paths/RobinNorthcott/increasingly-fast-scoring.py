# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import logging
from math import sqrt
from os.path import isfile
from os import system
from time import time

from numpy import zeros, argwhere, arange, roll, isin, logical_and, logical_not, squeeze, where
from scipy.sparse import load_npz, lil_matrix, save_npz

logging.basicConfig(level=logging.INFO)


def primes_sieve(limit):
    a = [True] * limit
    a[0] = a[1] = False
    out = []
    for (i, is_prime) in enumerate(a):
        if is_prime:
            for n in range(i * i, limit, i):
                a[n] = False
            out.append(i)
    return out


class Node:
    def __init__(self, id, x, y, is_prime):
        self.id = id
        self.x = x
        self.y = y
        self.is_prime = is_prime
        self.label = self.id
        self.neighbours = []
        self.f_max = None
        self.f_min = None
        self.bandwidth = None
        self.critical_nodes = []

    def update(self):
        labels = zeros(len(self.neighbours), dtype=int)
        for i, node in enumerate(self.neighbours):
            labels[i] = node.label
        self.f_max = max(labels)
        self.f_min = min(labels)
        bandwidth = abs(labels - self.label)
        self.bandwidth = max(bandwidth)
        self.critical_nodes = [self.neighbours[i] for i in argwhere(bandwidth == self.bandwidth)[0]]

    def set_label(self, label):
        self.label = label
        for node in self.neighbours:
            node.update()
        self.update()

    @staticmethod
    def dist(a: "Node", b: "Node"):
        return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


class CityMap:
    def __init__(self,
                 data_path='../input/cities.csv',
                 dist_matrix_path='./dist_matrix.matrix',):
        self.dist_matrix_path = dist_matrix_path
        self.cities = []
        with open(data_path, 'r') as file:
            lines = file.readlines()
        lines.pop(0)
        self.primes = primes_sieve(len(lines))
        for id_, line in enumerate(lines):
            line = line.split(',')
            self.cities.append(Node(int(line[0]), float(line[1]), float(line[2]), id_ in self.primes))
        self.n_cities = len(self.cities)
        self.distance_matrix = lil_matrix((self.n_cities, self.n_cities))

    def get_adjacency_list(self):
        return [[city_j.id for city_j in city_i.neighbours] for city_i in self.cities]

    def score_route(self, route=None):
        logging.info(f"scoring route")
        if route is None:
            route = arange(self.n_cities)
        distances = squeeze(self.distance_matrix[route, roll(route, -1)].toarray())
        missing = where(logical_not(distances.astype(bool)))[0]
        if len(missing):
            logging.info(f"{len(missing)} distances not found in matrix")
            for idx in missing:
                distances[idx] = Node.dist(self.cities[route[idx]], self.cities[route[(idx + 1) % self.n_cities]])
                self.distance_matrix[route[idx], route[(idx + 1) % self.n_cities]] = distances[idx]
                self.cities[route[idx]].neighbours.append(self.cities[route[(idx + 1) % self.n_cities]])
            save_npz(self.dist_matrix_path, self.distance_matrix.tocsr())
        mod_ten = logical_not((arange(self.n_cities) + 1) % 10)
        prime = isin(route, self.primes)
        factors = (logical_and(mod_ten, logical_not(prime)) * 0.1) + 1.0
        factors[0] = 1.0
        return factors * distances


print(sum(CityMap().score_route()))

