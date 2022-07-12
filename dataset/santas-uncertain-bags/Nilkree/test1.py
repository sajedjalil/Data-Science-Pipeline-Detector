from itertools import product, combinations_with_replacement
import numpy as np
import pandas as pd
import csv

def calc_weight(np_array):
    return float(np.std(np_array) + np.mean(np_array))


def horse_sample(size=1):
    return np.maximum(0, np.random.normal(5, 2, size))


def bike_sample(size=1):
    return np.maximum(0, np.random.normal(20, 10, size))


def ball_sample(size=1):
    return np.maximum(0, np.random.normal(1, 0.3, size))


def train_sample(size=1):
    return np.maximum(0, np.random.normal(10, 5, size))


def coal_sample(size=1):
    return 47 * np.random.beta(0.5, 0.5, size)


def book_sample(size=1):
    return np.random.chisquare(2, size)


def doll_sample(size=1):
    return np.random.gamma(5, 1, size)


def blocks_sample(size=1):
    return np.random.triangular(5, 10, 20, size)


def gloves_sample(size=1):
    dist1 = 3.0 + np.random.rand(size)
    dist2 = np.random.rand(size)
    toggle = np.random.rand(size) < 0.3
    dist2[toggle] = dist1[toggle]
    return dist2


GIFTS_SAMPLES = {'horse': horse_sample,
                 'ball': ball_sample,
                 'bike': bike_sample,
                 'train': train_sample,
                 'coal': coal_sample,
                 'book': book_sample,
                 'doll': doll_sample,
                 'blocks': blocks_sample,
                 'gloves': gloves_sample
                 }


def sample_weight(gift, quantity=1, size=1):
    return np.sum(GIFTS_SAMPLES[gift](quantity * size).reshape(quantity, size), axis=0)


def get_gift_weight(gift):
    array = sample_weight(gift, quantity=1, size=10000)
    return calc_weight(array)

# GIFTS = {'horse': 15,
#          'coal': 7,
#          'ball': 30,
#          'block': 5}

# WEIGHTS = {'horse': 2,
#            'coal': 8,
#            'ball': 1,
#            'block': 6}

MAX_WEIGHT = 50
MIN_WEIGHT = 30
MIN_COUNT = 3
BAGS_COUNT = 10


def get_gifts_count():
    data = pd.read_csv('../input/gifts.csv')
    result = {}
    for gift in data['GiftId']:
        name = gift.split('_')[0]
        result[name] = result.get(name, 0) + 1
    return result


GIFTS = get_gifts_count()
GIFTS_NAME_LIST = sorted(GIFTS)
WEIGHTS = {gift: get_gift_weight(gift) for gift in GIFTS_NAME_LIST}


def max_amount_in_bag(name):
    return int(MAX_WEIGHT / WEIGHTS[name])

MAX_IN_ONE_BAG = {gift: max_amount_in_bag(gift) for gift in GIFTS}


def to_csv(data):
    with open('test.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        writer.writerow(['bags'])
        for bag in data:
            writer.writerow([bag])
            break

possible_bags = (row for row in product(*(range(MAX_IN_ONE_BAG[gift] + 1) for gift in GIFTS_NAME_LIST)) if
                 sum(row) >= MIN_COUNT)
to_csv(possible_bags)
# a = list(possible_bags)
# print('complete')
# print(len(a))