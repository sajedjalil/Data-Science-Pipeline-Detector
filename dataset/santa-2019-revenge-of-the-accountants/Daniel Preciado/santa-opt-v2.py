import numpy as np
import pandas as pd
from numba import njit
import itertools, random
from multiprocessing import Pool, cpu_count

fam = pd.read_csv('/kaggle/input/santa-2019-revenge-of-the-accountants/family_data.csv')
sub = pd.read_csv('/kaggle/input/santa-2019-revenge-of-the-accountants/sample_submission.csv')
fam = pd.merge(fam,sub, how='left', on='family_id')
choices = fam[['choice_'+str(i) for i in range(10)]].values
fam = fam[['n_people','assigned_day']].values

fam_costs = np.zeros((6000,101))
for f in range(6000):
    for d in range(1,101):
        l = list(choices[f])
        if d in l:
            if l.index(d) == 0:
                fam_costs[f,d] = 0
            elif l.index(d) == 1:
                fam_costs[f,d] = 50
            elif l.index(d) == 2:
                fam_costs[f,d] = 50 + 9 * fam[f,0]
            elif l.index(d) == 3:
                fam_costs[f,d] = 100 + 9 * fam[f,0]
            elif l.index(d) == 4:
                fam_costs[f,d] = 200 + 9 * fam[f,0]
            elif l.index(d) == 5:
                fam_costs[f,d] = 200 + 18 * fam[f,0]
            elif l.index(d) == 6:
                fam_costs[f,d] = 300 + 18 * fam[f,0]
            elif l.index(d) == 7:
                fam_costs[f,d] = 300 + 36 * fam[f,0]
            elif l.index(d) == 8:
                fam_costs[f,d] = 400 + 36 * fam[f,0]
            elif l.index(d) == 9:
                fam_costs[f,d] = 500 + 235 * fam[f,0]
        else:
            fam_costs[f,d] = 500 + 434 * fam[f,0]
            
@njit(fastmath=True)
def fclip(p,l=0.):
    for i in range(len(p)):
        if p[i]<l:
            p[i]=l
    return p

@njit(fastmath=True)
def cost_function(pred, p1=1_000_000_000, p2=4000):
    days = np.array(list(range(100,0,-1)))
    daily_occupancy = np.zeros(101)
    penalty = 0
    for i in range(6000):
        penalty += fam_costs[i,pred[i,1]]
        daily_occupancy[pred[i,1]] += pred[i,0]

    for v in daily_occupancy[1:]:
        if (v < 125) or (v >300):
            if v > 300:
                penalty += p1 + abs(v-300)*p2
            else:
                penalty += p1 + abs(v-125)*p2

    penalty += max(0, (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5))
    
    for i in range(1,6):
        do = daily_occupancy[::-1] #reverse
        p = (do[i:] - 125.) / 400. * ((do[i:] ** (0.5 + ( np.abs(do[i:]-do[:-i]) / 50.0))) /(i**2))
        penalty += np.sum(fclip(p))

    return penalty

@njit(fastmath=True)
def the_shuffle(s=10):
    np.random.seed(s)
    l = np.arange(6000)
    np.random.shuffle(l)
    return l

@njit(fastmath=True)
def optimizer(pred):
    days = np.array(list(range(100,1,-1)))
    days_count = np.zeros(101)
    for i in range(6000):
        days_count[pred[i,1]] += pred[i,0]
    for f in the_shuffle():
        cd = int(pred[f,1])
        if cd > 1 and cd < 100:
            cp = int(pred[f,0])
            for d in days[1:-1]:
                if d != cd:
                    if days_count[d]+cp>=125 and days_count[d]+cp<=300 and days_count[cd]-cp >= 125 and days_count[cd]-cp<=300:
                        if fam_costs[f,d] <= fam_costs[f,cd]:
                            dtf = [fx for fx in range(6000) if ((pred[fx,1]==d) and (pred[fx,0]==cp))]
                            for j in dtf:
                                if j != f:
                                    if fam_costs[f,d] + fam_costs[j,cd] <= fam_costs[f,cd] + fam_costs[j,d]:
                                        pred[f,1] = int(d)
                                        pred[j,1] = int(cd)
                                        cd = int(d)
    return pred

@njit(fastmath=True)
def optimizer_a3(fam, p1=1_000_000_000, p2=4000, s=10):
    for f1 in the_shuffle(s):
        for d in range(1,101):
            temp = fam.copy()
            temp[f1,1] = d
            if cost_function(temp,p1,p2) < cost_function(fam,p1,p2):
                fam = temp.copy()
    return fam

@njit(fastmath=True)
def ftest(fam, days_count, t):
    for fc in range(10):
        for f in range(5999,-1,-1): 
            if fam[f,1] == 0:
                for d in range(1,101):
                    if d == choices[f,fc] and days_count[d]+fam[f,0]<=t: 
                        fam[f,1] = d
                        days_count[d] += fam[f,0]
    return fam, days_count

days_count = np.zeros(101)
for f in range(6000):
    fam[f,1] = 0
for end in range(7,255):
    fam, days_count = ftest(fam, days_count, end)
    
for f in range(6000):
    if fam[f,1] == 0:
        fam[f,1] = 50

best_fam = fam.copy()
best = cost_function(fam)
for p1 in [90,70,80,90]:
    for j in range(3):
        fam = optimizer_a3(fam,p1,10, j)
        fam = optimizer_a3(fam,100,20, j+1)
        fam = optimizer_a3(fam,100,100, j+2)
        fam = optimizer_a3(fam)
        fam = optimizer(fam)
        new = cost_function(fam)
        print(p1, j, new, new - best)
        if new < best:
            best = new
            best_fam = fam.copy()
            pd.DataFrame({'family_id':list(range(6000)), 'assigned_day':fam[:,1]}).to_csv(f'submission_{best}.csv', index=False)