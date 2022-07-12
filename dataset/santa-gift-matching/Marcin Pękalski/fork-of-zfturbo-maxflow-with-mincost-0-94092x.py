# coding: utf-8
# Kaggle dosn't have ortools installed. So you need to run kernel at local machine

__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'
import pip

def install(package):
    pip.main(['install', ortools])


import numpy as np
import pandas as pd
from ortools.graph import pywrapgraph

INPUT_PATH = '../input/'


def avg_normalized_happiness(pred, gift, wish):
    n_children = 1000000  # n children to give
    n_gift_type = 1000  # n types of gifts available
    twins = int(0.004 * n_children)  # 0.4% of all population, rounded to the closest even number

    # check if twins have the same gift
    for t1 in range(0, twins, 2):
        twin1 = pred[t1]
        twin2 = pred[t1 + 1]
        assert twin1 == twin2

    total_child_happiness = 0
    total_gift_happiness = 0

    for i in range(len(pred)):
        child_id = i
        gift_id = pred[i]

        # check if child_id and gift_id exist
        assert 0 <= child_id < n_children
        assert 0 <= gift_id < n_gift_type
        child_happiness = 20 - 2 * np.where(wish[child_id] == gift_id)[0]
        if not child_happiness:
            child_happiness = -1

        gift_happiness = 2000 - 2 * np.where(gift[gift_id] == child_id)[0]
        if not gift_happiness:
            gift_happiness = -1

        total_child_happiness += child_happiness
        total_gift_happiness += gift_happiness

    # print(max_child_happiness, max_gift_happiness
    child_h = float(total_child_happiness) / 20000000
    santa_h = float(total_gift_happiness) / 2000000000
    print('Normalized child happiness: ', child_h)
    print('Normalized santa happiness: ', santa_h)
    return child_h + santa_h


def get_overall_hapiness(wish, gift):

    res_child = dict()
    for i in range(0, 4000, 2):
        for j in range(wish.shape[1]):
            res_child[(i, wish[i][j])] = 50*(1 + (wish.shape[1] - j) * 2)
            res_child[(i + 1, wish[i][j])] = 50*(1 + (wish.shape[1] - j) * 2)
        for j in range(wish.shape[1]):
            if (i, wish[i+1][j]) in res_child:
                res_child[(i, wish[i+1][j])] += 50*(1 + (wish.shape[1] - j) * 2)
            else:
                res_child[(i, wish[i + 1][j])] = 50*(1 + (wish.shape[1] - j) * 2)
            if (i+1, wish[i + 1][j]) in res_child:
                res_child[(i + 1, wish[i+1][j])] += 50*(1 + (wish.shape[1] - j) * 2)
            else:
                res_child[(i + 1, wish[i + 1][j])] = 50*(1 + (wish.shape[1] - j) * 2)

    for i in range(4000, wish.shape[0]):
        for j in range(wish.shape[1]):
            res_child[(i, wish[i][j])] = 100*(1 + (wish.shape[1] - j)*2)

    res_santa = dict()
    for i in range(gift.shape[0]):
        for j in range(gift.shape[1]):
            res_santa[(gift[i][j], i)] = (1 + (gift.shape[1] - j)*2)

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


def solve():
    wish = pd.read_csv(INPUT_PATH + 'child_wishlist.csv', header=None).as_matrix()[:, 1:]
    gift_init = pd.read_csv(INPUT_PATH + 'gift_goodkids.csv', header=None).as_matrix()[:, 1:]
    gift = gift_init.copy()
    answ = np.zeros(len(wish), dtype=np.int32)
    answ[:] = -1
    gift_count = np.zeros(len(gift), dtype=np.int32)
    happiness = get_overall_hapiness(wish, gift)

    start_nodes = []
    end_nodes = []
    capacities = []
    unit_costs = []
    supplies = []

    for h in happiness:
        c, g = h
        # print(c, g, happiness[h])
        start_nodes.append(int(c))
        end_nodes.append(int(1000000 + g))
        capacities.append(1)
        unit_costs.append(-happiness[h])

    for i in range(1000000):
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
    print('Assigned: {}'.format(total))

    # instead of improving twins only by checking the first one, check both and 
    # compare the happiness score
    for i in range(0, 4000, 2):
        print('Improve twin {}'.format(i))
        if answ[i] != answ[i+1]:
            less_hapiness = 1000000000
            worst_child = [-1,-1]
            for k in [i, i+1]:
                for j in range(4000, 1000000):
                    if answ[j] == answ[k]:
                        if (j, answ[k]) in happiness:
                            score = happiness[(j, answ[k])]
                            if score < less_hapiness:
                                less_hapiness = score
                                if k == i:
                                    worst_child[0] = j
                                else:
                                    worst_child[1] = j
                        else:
                            if k == i:
                                worst_child[0] = j
                            else:
                                worst_child[1] = j
                            break
            if   0 < worst_child[0] and (worst_child[1] == -1 or worst_child[0] < worst_child[1]):
                    answ[worst_child[0]] = answ[i+1]
                    answ[i+1] = answ[i]
            elif 0 < worst_child[1] and (worst_child[0] == -1 or worst_child[1] < worst_child[0]):
                    answ[worst_child[1]] = answ[i]
                    answ[i] = answ[i+1]
    
    if answ.min() == -1:
        print('Some children without present')
        exit()

    if gift_count.max() > 1000:
        print('Some error in kernel: {}'.format(gift_count.max()))
        exit()

    print('Start score calculation...')
    score = avg_normalized_happiness(answ, gift_init, wish)
    print('Predicted score: {:.8f}'.format(score))

    out = open('subm_{}.csv'.format(score), 'w')
    out.write('ChildId,GiftId\n')
    for i in range(len(answ)):
        out.write(str(i) + ',' + str(answ[i]) + '\n')
    out.close()


if __name__ == '__main__':
    solve()