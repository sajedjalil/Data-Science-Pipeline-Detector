# coding: utf-8
# Kaggle dosn't have ortools installed. So you need to run kernel at local machine
# update from https://www.kaggle.com/zfturbo/max-flow-with-min-cost-in-10-minutes-0-9408

import numpy as np
import pandas as pd
from ortools.graph import pywrapgraph

INPUT_PATH = '../input/'

num_child = 1000000
num_prize = 1001
child_pref = 10
prize_pref = 1000
padding = num_child
source = num_child + num_prize
sink = source+1

wish = pd.read_csv(INPUT_PATH + 'child_wishlist.csv', header=None).as_matrix()[:, 1:]
gift_init = pd.read_csv(INPUT_PATH + 'gift_goodkids.csv', header=None).as_matrix()[:, 1:]
gift = gift_init.copy()
answ = np.zeros(len(wish), dtype=np.int32)
answ[:] = -1
gift_count = np.zeros(num_prize, dtype=np.int32)

edgeMap = dict()
for i in range(wish.shape[0]):
    for j in range(wish.shape[1]):
        edgeMap[(i, wish[i][j])] = 1000*(1 + (wish.shape[1] - j)*2)

for i in range(gift.shape[0]):
    for j in range(gift.shape[1]):
        if (gift[i][j], i) in edgeMap:
            edgeMap[(gift[i][j], i)] += 10*(1 + (gift.shape[1] - j)*2)
        else:
            edgeMap[(gift[i][j], i)] = 10*(1 + (gift.shape[1] - j)*2)

start_nodes = []
end_nodes = []
capacities = []
unit_costs = []
supplies = []

for h in edgeMap:
    c, g = h
    # print(c, g, edgeMap[h])
    start_nodes.append(int(padding + g))
    end_nodes.append(int(c))
    capacities.append(1)
    unit_costs.append(41010-edgeMap[h])

# Instantiate a SimpleMinCostFlow solver.
min_cost_flow = pywrapgraph.SimpleMinCostFlow()

# Add each arc.
for i in range(0, len(start_nodes)):
    min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i], capacities[i], unit_costs[i])
for i in range(min(num_prize,1000)):
    min_cost_flow.AddArcWithCapacityAndUnitCost(source, i+padding, 1000, 0)
for i in range(num_child):
    min_cost_flow.AddArcWithCapacityAndUnitCost(1000+padding, i, 1, 41010)
min_cost_flow.AddArcWithCapacityAndUnitCost(source, 1000+padding, num_child, 0)

# Add node supplies.
min_cost_flow.SetNodeSupply(source, num_child)
for i in range(num_child):
    min_cost_flow.SetNodeSupply(i, -1)

# Find the minimum cost flow
print('Start solve....')
min_cost_flow.SolveMaxFlowWithMinCost()
res1 = min_cost_flow.MaximumFlow()
print('Maximum flow:', res1)
res2 = min_cost_flow.OptimalCost()
print('Optimal cost:', -res2 / 2e10)
print('Num arcs:', min_cost_flow.NumArcs())

total = 0
for i in range(min_cost_flow.NumArcs()):
    if min_cost_flow.Flow(i) == 1 and min_cost_flow.Head(i) >= 0 and min_cost_flow.Head(i) < num_child:
        answ[min_cost_flow.Head(i)] = min_cost_flow.Tail(i) - padding
        gift_count[min_cost_flow.Tail(i) - padding] += 1
print('Assigned: {}'.format(total))

for i in range(num_prize):
    if(gift_count[i] != 1000):
        print("prize=", gift_count[i],": ", i)
        
for i in range(num_child):
    if answ[i] == 1000:
        for j in range(min(num_prize,1000)):
            if gift_count[j] < 1000:
                answ[i] = j
                gift_count[j] += 1
                break

df = pd.read_csv(INPUT_PATH+"sample_submission_random.csv")
df['GiftId'] = answ
df.sort_values(['GiftId'], ascending=[1]).to_csv('sub.csv', index=False)