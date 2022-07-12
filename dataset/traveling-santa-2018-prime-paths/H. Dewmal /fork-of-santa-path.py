import sys

import pandas as pd
import numpy as np
import swifter
import math
import os
import pickle
import gzip
from operator import itemgetter
import matplotlib.pyplot as plt

import multiprocessing as mp

INPUT_PATH = '../input/'
OUTPUT_PATH = './'


def save_in_file_fast(arr, file_name):
    pickle.dump(arr, open(file_name, 'wb'))


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))


def SieveOfEratosthenes(n):
    # Create a boolean array "prime[0..n]" and initialize
    #  all entries it as true. A value in prime[i] will
    # finally be false if i is Not a prime, else true.
    prime = [True for i in range(n + 1)]
    prime[0] = False
    prime[1] = False
    p = 2
    while (p * p <= n):

        # If prime[p] is not changed, then it is a prime
        if (prime[p] == True):

            # Update all multiples of p
            for i in range(p * 2, n + 1, p):
                prime[i] = False
        p += 1
    return prime


def get_primes():
    cache_path = OUTPUT_PATH + 'prime_list.pkl'
    if not os.path.isfile(cache_path):
        n = 200000
        prime = SieveOfEratosthenes(n)
        plist = []
        for p in range(2, n):
            if prime[p]:
                plist.append(p)
        save_in_file_fast(set(plist), cache_path)
    else:
        plist = load_from_file_fast(cache_path)

    return plist


plist = get_primes()


def get_score(subm_path):
    cities = pd.read_csv(INPUT_PATH + 'cities.csv')
    all_ids = cities['CityId'].values
    all_x = cities['X'].values
    all_y = cities['Y'].values

    arr = dict()
    for i, id in enumerate(all_ids):
        arr[id] = (all_x[i], all_y[i])

    score = 0.0
    s = pd.read_csv(subm_path)['Path'].values
    for i in range(0, len(s) - 1):
        p1 = arr[s[i]]
        p2 = arr[s[i + 1]]
        stepSize = math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
        if ((i + 1) % 10 == 0) and (s[i] not in plist):
            stepSize *= 1.1
        # print(stepSize)
        score += stepSize
    return score


def euq_dist(x, y, x_adjust=0.0, y_adjust=0.0):
    x = x - x_adjust  # 316.836739061509
    y = y - y_adjust  # 2202.34070733524
    return math.sqrt(math.pow(x, 2) + math.pow(y, 2))


city_0_dist = euq_dist(316.836739061509, 2202.34070733524)


# 1513747.36
# 130635222.73
# 1812602
# 73929960


def calculate_id(row):
    x = float(row['X'])
    y = float(row['Y'])
    return euq_dist(x, y, x_adjust=316.836739061509, y_adjust=2202.34070733524)


def calculate_id_2(row):
    x = float(row['X'])
    y = float(row['Y'])
    return euq_dist(x, y)


cities = pd.read_csv(INPUT_PATH + 'cities.csv')
cities['idx'] = cities.swifter.apply(calculate_id, axis=1)

cities = cities.sort_values(by=['idx'])
cities = cities.reset_index(drop=True)


def calculate_id_3(row, consider_value=100):
    # print(len(row))
    id = int(row.name)
    center_x = row.X
    center_y = row.Y
    vals = cities.iloc[id:id + consider_value]
    # print("---", len(vals))
    min_id = id
    min_val = 999999999
    idx = 0
    for val in vals.values:

        # print(val)
        x = val[1]
        y = val[2]

        dist = euq_dist(x, y, x_adjust=center_x, y_adjust=center_y)
        _id = id + idx
        if (_id % 10 == 0) and (_id not in plist):
            dist *= 1.1

        if dist > 0.0:
            if min_val > dist:
                min_val = dist
                min_id = _id

        # print(id + idx, dist)
        idx += 1

    # print("---", min_id)
    return min_id


considers = [100]

epochs = 25
for epoch in range(epochs):
    print(f'Start Epoch = {epoch}/{epochs}')
    for consider_value in considers:
        cities['final_path_value'] = cities.swifter.apply(
            lambda row: calculate_id_3(row, consider_value=consider_value), axis=1)
        cities = cities.sort_values(by=['final_path_value'])
        cities = cities.reset_index(drop=True)
        print(f'Done at {consider_value}')
    print(f'End Epoch = {epoch}/{epochs}')

print(cities.head())

cities.head()

# plt.scatter(cities.X.values, cities.Y.values)
# plt.scatter([316.836739061509], [2202.34070733524])
# # plt.scatter([], [])
# plt.show()

cities_ids = cities.CityId.values

#
cities_id_results = np.concatenate((cities_ids, [0]), axis=None)
# cities_id_results
d = {'Path': cities_id_results}

df = pd.DataFrame(data=d)
df.to_csv( 'sample_submission.csv', index=False)

if __name__ == '__main__':
    score = get_score( 'sample_submission.csv')
    print('Score: {:.2f}'.format(score))
