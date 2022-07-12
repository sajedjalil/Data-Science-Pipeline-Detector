# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import pandas as pd
import numpy as np
from collections import Counter
import operator
import math


INPUT_PATH = '../input/'


def lcm(a, b):
    """Compute the lowest common multiple of a and b"""
    # in case of large numbers, using floor division
    return a * b // math.gcd(a, b)


def avg_normalized_happiness(pred, gift, wish):
    
    n_children = 1000000 # n children to give
    n_gift_type = 1000 # n types of gifts available
    n_gift_quantity = 1000 # each type of gifts are limited to this quantity
    n_gift_pref = 100 # number of gifts a child ranks
    n_child_pref = 1000 # number of children a gift ranks
    twins = math.ceil(0.04 * n_children / 2.) * 2    # 4% of all population, rounded to the closest number
    triplets = math.ceil(0.005 * n_children / 3.) * 3    # 0.5% of all population, rounded to the closest number
    ratio_gift_happiness = 2
    ratio_child_happiness = 2

    # check if triplets have the same gift
    for t1 in np.arange(0, triplets, 3):
        triplet1 = pred[t1]
        triplet2 = pred[t1+1]
        triplet3 = pred[t1+2]
        # print(t1, triplet1, triplet2, triplet3)
        assert triplet1 == triplet2 and triplet2 == triplet3
                
    # check if twins have the same gift
    for t1 in np.arange(triplets, triplets+twins, 2):
        twin1 = pred[t1]
        twin2 = pred[t1+1]
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
        child_happiness = (n_gift_pref - np.where(wish[child_id]==gift_id)[0]) * ratio_child_happiness
        if not child_happiness:
            child_happiness = -1

        gift_happiness = ( n_child_pref - np.where(gift[gift_id]==child_id)[0]) * ratio_gift_happiness
        if not gift_happiness:
            gift_happiness = -1

        total_child_happiness += child_happiness
        total_gift_happiness[gift_id] += gift_happiness
        
    denominator1 = n_children*max_child_happiness
    denominator2 = n_gift_quantity*max_gift_happiness*n_gift_type
    common_denom = lcm(denominator1, denominator2)
    multiplier = common_denom / denominator1

    ret = float(math.pow(total_child_happiness*multiplier,3) + \
        math.pow(np.sum(total_gift_happiness),3)) / float(math.pow(common_denom,3))
    return ret
    

def get_overall_hapiness(wish, gift):


    res_child = dict()
    for i in range(0, wish.shape[0]):
        for j in range(55):
            res_child[(i, wish[i][j])] = int(100* (1 + (wish.shape[1] - j)*2))

    res_santa = dict()
    for i in range(gift.shape[0]):
        for j in range(gift.shape[1]):
            res_santa[(gift[i][j], i)] = int((1 + (gift.shape[1] - j)*2))

    positive_cases = list(set(res_santa.keys()) | set(res_child.keys()))
    print('Positive case tuples (child, gift): {}'.format(len(positive_cases)))

    res = dict()
    for p in positive_cases:
        res[p] = 0
        if p in res_child:
            res[p] += res_child[p]
        if p in res_santa:
            res[p] += res_santa[p]
    return res


def sort_dict_by_values(a, reverse=True):
    sorted_x = sorted(a.items(), key=operator.itemgetter(1), reverse=reverse)
    return sorted_x


def value_counts_for_list(lst):
    a = dict(Counter(lst))
    a = sort_dict_by_values(a, True)
    return a


def get_most_desired_gifts(wish, gift):
    best_gifts = value_counts_for_list(np.ravel(wish))
    return best_gifts


def recalc_hapiness(happiness, best_gifts, gift):
    recalc = dict()
    for b in best_gifts:
        recalc[b[0]] = b[1] / 2000000

    for h in happiness:
        c, g = h
        happiness[h] /= recalc[g]

        # Make triples/twins more happy
        # if c <= 45000 and happiness[h] < 0.00001:
        #     happiness[h] = 0.00001

    return happiness


def solve():
    wish = pd.read_csv(INPUT_PATH + 'child_wishlist_v2.csv', header=None).as_matrix()[:, 1:]
    gift_init = pd.read_csv(INPUT_PATH + 'gift_goodkids_v2.csv', header=None).as_matrix()[:, 1:]
    gift = gift_init.copy()
    answ = np.zeros(len(wish), dtype=np.int32)
    answ[:] = -1
    gift_count = np.zeros(len(gift), dtype=np.int32)

    happiness = get_overall_hapiness(wish, gift)
    best_gifts = get_most_desired_gifts(wish, gift)
    happiness = recalc_hapiness(happiness, best_gifts, gift)
    sorted_hapiness = sort_dict_by_values(happiness)
    print('Happiness sorted...')

    for i in range(len(sorted_hapiness)):
        child = sorted_hapiness[i][0][0]
        g = sorted_hapiness[i][0][1]
        if answ[child] != -1:
            continue
        if gift_count[g] >= 1000:
            continue
        if child <= 5000 and gift_count[g] < 997:
            if child % 3 == 0:
                answ[child] = g
                answ[child+1] = g
                answ[child+2] = g
                gift_count[g] += 3
            elif child % 3 == 1:
                answ[child] = g
                answ[child-1] = g
                answ[child+1] = g
                gift_count[g] += 3
            else:
                answ[child] = g
                answ[child-1] = g
                answ[child-2] = g
                gift_count[g] += 3
        elif child > 5000 and child <= 45000 and gift_count[g] < 998:
            if child % 2 == 0:
                answ[child] = g
                answ[child - 1] = g
                gift_count[g] += 2
            else:
                answ[child] = g
                answ[child + 1] = g
                gift_count[g] += 2
        elif child > 45000:
            answ[child] = g
            gift_count[g] += 1

    print('Left unhappy:', len(answ[answ == -1]))
    
    # unhappy children
    for child in range(45001, len(answ)):
        if answ[child] == -1:
            g = np.argmin(gift_count)
            answ[child] = g
            gift_count[g] += 1

    if answ.min() == -1:
        print('Some children without present')
        exit()

    if gift_count.max() > 1000:
        print('Some error in kernel: {}'.format(gift_count.max()))
        exit()

    print('Start score calculation...')
    # score = avg_normalized_happiness(answ, gift_init, wish)
    # print('Predicted score: {:.8f}'.format(score))
    score = avg_normalized_happiness(answ, gift, wish)
    print('Predicted score: {:.8f}'.format(score))

    out = open('subm_{}.csv'.format(score), 'w')
    out.write('ChildId,GiftId\n')
    for i in range(len(answ)):
        out.write(str(i) + ',' + str(answ[i]) + '\n')
    out.close()


if __name__ == '__main__':
    solve()