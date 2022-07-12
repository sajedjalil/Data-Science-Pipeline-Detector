# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import numpy as np
import pandas as pd
import math
import gc
from ortools.graph import pywrapgraph


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

    print(multiplier, common_denom)
    child_hapiness = math.pow(total_child_happiness * multiplier, 3) / float(math.pow(common_denom, 3))
    santa_hapiness = math.pow(np.sum(total_gift_happiness), 3) / float(math.pow(common_denom, 3))
    print('Child hapiness: {}'.format(child_hapiness))
    print('Santa hapiness: {}'.format(santa_hapiness))
    ret = child_hapiness + santa_hapiness
    return ret


def get_overall_hapiness(wish, gift):

    list_limit = wish.shape[1]
    list_limit = 42

    res_child = dict()
    for i in range(0, 5001):
        app = i - (i % 3)
        for j in range(list_limit):
            if (app, wish[i][j]) in res_child:
                res_child[(app, wish[i][j])] += 10 * (1 + (wish.shape[1] - j) * 2)
            else:
                res_child[(app, wish[i][j])]  = 10 * (1 + (wish.shape[1] - j) * 2)

    for i in range(5001, 45001):
        app = i + (i % 2)
        for j in range(list_limit):
            if (app, wish[i][j]) in res_child:
                res_child[(app, wish[i][j])] += 10 * (1 + (wish.shape[1] - j) * 2)
            else:
                res_child[(app, wish[i][j])]  = 10 * (1 + (wish.shape[1] - j) * 2)

    for i in range(45001, wish.shape[0]):
        app = i
        for j in range(list_limit):
            res_child[(app, wish[i][j])]  = 10 * (1 + (wish.shape[1] - j) * 2)

    res_santa = dict()
    for i in range(gift.shape[0]):
        for j in range(gift.shape[1]):
            cur_child = gift[i][j]
            if cur_child < 5001:
                cur_child -= cur_child % 3
            elif cur_child < 45001:
                cur_child += cur_child % 2
            res_santa[(cur_child, i)] = (1 + (gift.shape[1] - j)*2)

    positive_cases = list(set(res_santa.keys()) | set(res_child.keys()))
    print('Positive case tuples (child, gift): {}'.format(len(positive_cases)))

    res = dict()
    for p in positive_cases:
        res[p] = 0
        if p in res_child:
            a = res_child[p]
            res[p] += int((a ** 3) * 4)
        if p in res_santa:
            b = res_santa[p]
            res[p] += int((b ** 3) / 4)

    return res


def solve():
    wish = pd.read_csv(INPUT_PATH + 'child_wishlist_v2.csv', header=None).as_matrix()[:, 1:]
    gift = pd.read_csv(INPUT_PATH + 'gift_goodkids_v2.csv', header=None).as_matrix()[:, 1:]
    answ = np.zeros(len(wish), dtype=np.int32)
    answ[:] = -1
    happiness = get_overall_hapiness(wish, gift)
    gc.collect()

    start_nodes = []
    end_nodes = []
    capacities = []
    unit_costs = []
    supplies = []

    min_h = 10**100
    max_h = -10**100
    avg_h = 0
    for h in happiness:
        c, g = h

        start_nodes.append(int(c))
        end_nodes.append(int(1000000 + g))
        if c < 5001:
            capacities.append(3)
        elif c < 45001:
            capacities.append(2)
        else:
            capacities.append(1)
        unit_costs.append(-happiness[h])
        if happiness[h] > max_h:
            max_h = happiness[h]
        if happiness[h] < min_h:
            min_h = happiness[h]
        avg_h += happiness[h]
    print('Max single happiness: {}'.format(max_h))
    print('Min single happiness: {}'.format(min_h))
    print('Avg single happiness: {}'.format(avg_h / len(happiness)))

    for i in range(1000000):
        if i < 5001:
            supplies.append(3)
        elif i < 45001:
            supplies.append(2)
        else:
            supplies.append(1)
    for j in range(1000000, 1001000):
        supplies.append(-1000)

    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    # Add each arc.
    for i in range(0, len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i], capacities[i], unit_costs[i])

    # Add node supplies.
    for i in range(0, len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])

    # Find the minimum cost flow
    print('Start solve....')
    min_cost_flow.SolveMaxFlowWithMinCost()
    res1 = min_cost_flow.MaximumFlow()
    print('Maximum flow:', res1)
    res2 = min_cost_flow.OptimalCost()
    print('Optimal cost:', -res2 / 2000000000)
    print('Num arcs:', min_cost_flow.NumArcs())

    total = 0
    for i in range(min_cost_flow.NumArcs()):
        cost = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
        if cost != 0:
            answ[min_cost_flow.Tail(i)] = min_cost_flow.Head(i) - 1000000
            total += 1
    print('Assigned: {}'.format(total))

    print('Check for overflow...')
    gift_count = np.zeros(1000, dtype=np.int32)
    for i in range(len(answ)):
        if answ[i] != -1:
            gift_count[answ[i]] += 1
    for i in range(1000):
        if gift_count[i] > 1000:
            print('Gift error: {} (Value: {})'.format(i, gift_count[i]))

    # Add triplets restrictions
    for i in range(0, 5001, 3):
        answ[i + 1] = answ[i]
        answ[i + 2] = answ[i]

    # Add twins restrictions
    for i in range(5001, 45001, 2):
        answ[i] = answ[i + 1]

    if answ.min() == -1:
        print('Some children without present')
        exit()

    print('Check for overflow after twins/triplets assigned')
    gift_count = np.zeros(1000, dtype=np.int32)
    for i in range(len(answ)):
        gift_count[answ[i]] += 1

    ov_count = 0
    for i in range(1000):
        if gift_count[i] > 1000:
            # print('Gift error: {} (Value: {})'.format(i, gift_count[i]))
            ov_count += 1
    if gift_count.max() > 1000:
        print('Gift overflow! Count: {}'.format(ov_count))

    for i in range(45001, len(answ)):
        if gift_count[answ[i]] > 1000:
            old_val = answ[i]
            j = np.argmin(gift_count)
            answ[i] = j
            gift_count[old_val] -= 1
            gift_count[j] += 1

    print('Check for overflow after simple fix')
    gift_count = np.zeros(1000, dtype=np.int32)
    for i in range(len(answ)):
        gift_count[answ[i]] += 1

    ov_count = 0
    for i in range(1000):
        if gift_count[i] > 1000:
            print('Gift error: {} (Value: {})'.format(i, gift_count[i]))
            ov_count += 1
    if gift_count.max() > 1000:
        print('Gift overflow! Count: {}'.format(ov_count))
        exit()

    print('Start score calculation...')
    score = avg_normalized_happiness(answ, gift, wish)
    print('Predicted score: {:.12f}'.format(score))

    out = open('subm_{:.12f}.csv'.format(score), 'w')
    out.write('ChildId,GiftId\n')
    for i in range(len(answ)):
        out.write(str(i) + ',' + str(answ[i]) + '\n')
    out.close()


if __name__ == '__main__':
    solve()
