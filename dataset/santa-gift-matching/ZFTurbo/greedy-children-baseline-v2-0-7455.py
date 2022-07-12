# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import pandas as pd
import numpy as np
import math

INPUT_PATH = '../input/'


def lcm(a, b):
    """Compute the lowest common multiple of a and b"""
    # in case of large numbers, using floor division
    return a * b // math.gcd(a, b)


def avg_normalized_happiness(pred, gift, wish):
    n_children = 1000000  # n children to give
    n_gift_type = 1000  # n types of gifts available
    n_gift_quantity = 1000  # each type of gifts are limited to this quantity
    n_gift_pref = 100  # number of gifts a child ranks
    n_child_pref = 1000  # number of children a gift ranks
    twins = math.ceil(0.04 * n_children / 2.) * 2  # 4% of all population, rounded to the closest number
    triplets = math.ceil(0.005 * n_children / 3.) * 3  # 0.5% of all population, rounded to the closest number
    ratio_gift_happiness = 2
    ratio_child_happiness = 2

    # check if triplets have the same gift
    for t1 in np.arange(0, triplets, 3):
        triplet1 = pred[t1]
        triplet2 = pred[t1 + 1]
        triplet3 = pred[t1 + 2]
        # print(t1, triplet1, triplet2, triplet3)
        assert triplet1 == triplet2 and triplet2 == triplet3

    # check if twins have the same gift
    for t1 in np.arange(triplets, triplets + twins, 2):
        twin1 = pred[t1]
        twin2 = pred[t1 + 1]
        # print(t1)
        assert twin1 == twin2

    max_child_happiness = n_gift_pref * ratio_child_happiness
    max_gift_happiness = n_child_pref * ratio_gift_happiness
    total_child_happiness = 0
    total_gift_happiness = np.zeros(n_gift_type)

    for i in range(len(pred)):
        child_id = i
        gift_id = pred[i]

        # check if child_id and gift_id exist
        assert child_id < n_children
        assert gift_id < n_gift_type
        assert child_id >= 0
        assert gift_id >= 0
        child_happiness = (n_gift_pref - np.where(wish[child_id] == gift_id)[0]) * ratio_child_happiness
        if not child_happiness:
            child_happiness = -1

        gift_happiness = (n_child_pref - np.where(gift[gift_id] == child_id)[0]) * ratio_gift_happiness
        if not gift_happiness:
            gift_happiness = -1

        total_child_happiness += child_happiness
        total_gift_happiness[gift_id] += gift_happiness

    denominator1 = n_children * max_child_happiness
    denominator2 = n_gift_quantity * max_gift_happiness * n_gift_type
    common_denom = lcm(denominator1, denominator2)
    multiplier = common_denom / denominator1

    ret = float(math.pow(total_child_happiness * multiplier, 3) + \
                math.pow(np.sum(total_gift_happiness), 3)) / float(math.pow(common_denom, 3))
    return ret
           
           
def solve():
    wish = pd.read_csv(INPUT_PATH + 'child_wishlist_v2.csv', header=None).as_matrix()[:, 1:]
    gift = pd.read_csv(INPUT_PATH + 'gift_goodkids_v2.csv', header=None).as_matrix()[:, 1:]
    answ = np.zeros((len(wish)), dtype=np.int32)
    answ[:] = -1
    gift_count = np.zeros((len(gift)), dtype=np.int32)

    for i in range(0, 5001, 3):
        g = wish[i, 0]
        answ[i] = g
        answ[i+1] = g
        answ[i+2] = g
        gift_count[g] += 3
    
    for i in range(5001, 45001, 2):
        g = wish[i, 0]
        answ[i] = g
        answ[i+1] = g
        gift_count[g] += 2

    for i in range(45001, len(answ)):
        for k in range(100):
            g = wish[i, k]
            if gift_count[g] < 1000:
                answ[i] = g
                gift_count[g] += 1
                break
        if answ[i] == -1:
            g = np.argmin(gift_count)
            answ[i] = g
            gift_count[g] += 1

        if i % 100000 == 0:
            print('Completed: {}'.format(i))

    if gift_count.max() > 1000:
        print('Some error in kernel: {}'.format(gift_count.max()))

    score = avg_normalized_happiness(answ, gift, wish)
    print('Predicted score: {:.8f}'.format(score))

    out = open('subm_{}.csv'.format(score), 'w')
    out.write('ChildId,GiftId\n')
    for i in range(len(answ)):
        out.write(str(i) + ',' + str(answ[i]) + '\n')
    out.close()


if __name__ == '__main__':
    solve()