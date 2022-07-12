# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


import pandas as pd
import numpy as np
import math
import os
import pickle
import gzip
from operator import itemgetter


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


def get_score(subm_path):
    plist = get_primes()
    cities = pd.read_csv(INPUT_PATH + 'cities.csv')
    all_ids = cities['CityId'].values
    all_x = cities['X'].values
    all_y = cities['Y'].values

    arr = dict()
    for i, id in enumerate(all_ids):
        arr[id] = (all_x[i], all_y[i])

    score = 0.0
    s = pd.read_csv(subm_path)['Path'].values
    for i in range(0, len(s)-1):
        p1 = arr[s[i]]
        p2 = arr[s[i+1]]
        stepSize = math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
        if ((i + 1) % 10 == 0) and (s[i] not in plist):
            stepSize *= 1.1
        # print(stepSize)
        score += stepSize
    return score
    

if __name__ == '__main__':
    score = get_score(INPUT_PATH + 'sample_submission.csv')
    print('Score: {:.2f}'.format(score))