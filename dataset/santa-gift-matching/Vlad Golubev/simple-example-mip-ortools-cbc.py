from __future__ import print_function
from ortools.linear_solver import pywraplp
from collections import defaultdict
import pandas as pd
import time
from tqdm import tqdm
n_children = 10000
n_gift_type = 100
n_gift_quantity = 100
n_gift_pref = 5
n_child_pref = 100
twins = 40
ratio_gift_happiness = 2
ratio_child_happiness = 2

solver = pywraplp.Solver('SmallInstance', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
# https://www.kaggle.com/batzner/small-dataset-generator/output
wish = pd.read_csv('input/child_wishlist_small.csv', header=None).as_matrix()[:, 1:]

x = {}
cost = defaultdict(int)

for i in tqdm(range(n_children)):
    for j in range(n_gift_pref):
        if i < 40:
            cost[i,  wish[i,j]] += 2 * (n_gift_pref - j)+1
            if i%2 == 1:
                cost[i-1,wish[i,j]] += 0
            else:
                cost[i+1,wish[i,j]] += 0
        else:
            cost[i,  wish[i,j]] += 2 * (n_gift_pref - j)+1
        
max_sum = [0 for i in range(len(cost))]
child_m = [[] for i in range(n_children)]
gift_m = [[] for i in range(n_gift_type)]
for i, h in enumerate(tqdm(cost)):
    cid, gid = h
    t = solver.BoolVar('')
    x[cid, gid] = t
    child_m[cid].append(t)
    gift_m[gid].append(t)
    max_sum[i] = cost[h] * t

solver.Maximize(solver.Sum(max_sum))

tt = 0
for i in tqdm(range(0, twins, 2)):
    for j in range(n_gift_quantity):
        if (i, j) in cost:
            tt += 1
            solver.Add(x[i,j] - x[i+1, j] == 0)

# Each child is assigned to at most 1 gift.
for cid in tqdm(range(n_children)):
    solver.Add(solver.Sum(child_m[cid]) == 1)
del child_m

# Each gift is assigned to at most some number of children.
for gid in tqdm(range(n_gift_type)):
    solver.Add(solver.Sum(gift_m[gid]) == n_gift_quantity)
del gift_m
print(tt)
start = time.time()
print('start solving')
sol = solver.Solve()
print(time.time() - start, 'sec')
print('Total cost = ', solver.Objective().Value())
print("Time = ", solver.WallTime(), " milliseconds")

for i in range(n_children):
    for j in range(n_gift_quantity+1):
        if (i,j) in cost and x[i, j].solution_value() > 0:
            print('Worker %d assigned to task %d.    Cost = %d' % (i,j,cost[i, j]))