import pandas as pd
from haversine import haversine
from copy import deepcopy
import time
t1 = time.time()
north_pole = (90,0)
weight_limit = 1000.0
gifts = pd.read_csv("../input/gifts.csv").fillna(" ")
### FOR TESTING PURPOSES ONLY ###
gifts = gifts
weight_limit = 100.0
### FOR TESTING PURPOSES ONLY ###
giftsNS = gifts.sort_values(by=['Latitude'], ascending=[1])
giftsWE = gifts.sort_values(by=['Longitude'], ascending=[1])
giftsWt = gifts.sort_values(by=['Weight'], ascending=[0])



def single_costs():
    costs = []
    for i in range(len(gifts)):
        costs.append(trip_cost([i]))
    return costs

def angular_dist(angle_a,angle_b):
    return abs((angle_a - angle_b + 180) % 360 - 180)

def coords(a):
    A = gifts[a:a+1]
    cA = (float(A['Latitude']),float(A['Longitude']))
    return cA

def weight(a):
    A = gifts[a:a+1]
    wA = float(A['Weight'])
    return wA

def row(i):
    R = gifts[i:i+1]
    return R
    
def dist(cA,cB):
    return haversine(cA,cB)

def trip_cost(L):
    wt = trip_weight(L)
    cost = wt*dist(north_pole,coords(L[0]))
    prevLoc = coords(L[0])
    for i,G in enumerate(L[1:]):
        wt -= weight(G)
        currLoc = coords(G)
        ds = dist(currLoc,prevLoc)
        cost += wt * ds
        #print(cost, wt*ds, ds, wt)
        prevLoc = currLoc
    return cost

def trip_length(L):
    d = 0
    prevLoc = north_pole
    for l in L:
        currLoc = coords(l)
        d += dist(currLoc,prevLoc)
        prevLoc = currLoc
    d += dist(north_pole, prevLoc)
    return d
    
def trip_weight(L):
    tw = 10
    for g in L:
        tw += weight(g)
    return tw

def min_weight(L):
    m = 2000
    for g in L:
        m = min(m, weight(g))
    return m


def add_to(L,g):
    new_cost = trip_cost(L)
    imin = -1
    Lp = deepcopy(L)
    for i in range(len(L)):
        Lp.insert(i,g)
        tw = trip_weight(Lp)
        if tw <= weight_limit:
            tc = trip_cost(Lp)
            if tc < new_cost:
                new_cost = tc
                imin = i
        Lp.pop(i)
    return new_cost,imin

def isnt_full(L,Remain):
    w = trip_weight(L)
    m = min_weight(Remain)
    #print(w,m)
    return w + m < weight_limit

def find_an_optimal_trip(gCol):
    gifts_by_weight = gifts[gCol].sort_values(by=['Weight'], ascending=[0])
    gifts_by_lat = gifts[gCol].sort_values(by=['Latitude'], ascending=[0])
    for i in gCol:
        pass
    return None
    
def timed_out():
    t2 = time.time()
    if t2-t1 > 120:
        return True
    else:
        return False
        
costs = single_costs()
#full_cost = sum(costs)
Full = []
Partial = [[0]]
All = range(len(gifts))
L = Partial[0]
Remain = [x for x in All if x not in L]
while Remain:
    while isnt_full(L,Remain) and not timed_out():
        costL = trip_cost(L)
        costR = 0
        for r in Remain:
            costR += costs[r]
        
        imin = None
        for g in Remain:
            nc, i = add_to(L,g)
            if nc < costL + costs[g]:
                gmin = g
                ncmin = nc
                imin = i
        if imin:
            L.insert(imin,gmin)
            Remain.remove(gmin)
        else:
            break
    print('trip: ',L,'but',len(Remain),'gifts still remain to be delivered')
    if Remain:
        L = [Remain.pop()]
    else:
        print('all done!')
'''
print(trip_cost(L))
print(trip_weight(L))
print(trip_length(L),trip_length(L[:-1]))
for l in L:
    print(coords(l))
#print(gifts[a:a+1])
#print(gifts[b:b+1])
#print(gifts[[3]])
'''