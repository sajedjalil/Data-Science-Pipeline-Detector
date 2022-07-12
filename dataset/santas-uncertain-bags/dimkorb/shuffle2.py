from sklearn.utils import shuffle
from random import randint
import pandas as pd
import numpy as np
import copy

bags = [[] for b in range(1000)]
for b in range(166):
    bags[b] = ['coal_'+str(b), 'doll_'+str(b), 'book_'+str(b)]
for b in range(166,566,1):
    bags[b] = ['blocks_'+str(b-166), 'horse_'+str(b-166), 'bike_'+str(b-166)]
for b in range(566,766,1):
    bags[b] = ['blocks_'+str(b-166), 'blocks_'+str(b-166+200), 'blocks_'+str(b-166+400), 'doll_'+str(b-400), 'gloves_'+str(b-566)]
i = 0
for b in range(766,1000,1):
    bags[b] = ['train_'+str(i)]
    for j in range(4):
        bags[b].append('ball_'+str(i*4+j))
        bags[b].append('book_'+str(i*4+j+166))
    for j in range(2):
        bags[b].append('doll_'+str(i*2+j+366))
    i+=1

def shuffle_function(xbags, xgift_list):
    xgift_list = shuffle(xgift_list)
    for b in range(len(xbags)):
        for g in range(len(xbags[b])):
            for j in range(len(xgift_list)):
                if xbags[b][g].split('_')[0] == xgift_list[j].split('_')[0]:
                    temp = copy.deepcopy(xbags[b])
                    temp[g] = str(xgift_list[j])
                    xgift_list[j] = str(xbags[b][g])
                    xbags[b] = copy.deepcopy(temp)
                    break
                elif xbags[b][g].split('_')[0] == 'coal':
                    break
        if b % 200 == 0:
            print('Swap with gift list', b)
    return xbags, xgift_list

gift_list = pd.read_csv('../input/gifts.csv')
gift_list = list(gift_list['GiftId'].values)
gifts_in_bags = []
for i in range(len(bags)):
    gifts_in_bags += bags[i]
gift_list = [x for x in gift_list if x not in gifts_in_bags]
gift_list = [x for x in gift_list if 'coal' not in x] #remove coal

for i in range(1):
    print(i, '='*20)
    bags, gift_list = shuffle_function(bags, gift_list)
    out = open('submission_fun' + str(i) + '.csv', 'w')
    out.write('Gifts\n')
    for b in bags:
        out.write(' '.join(b) + '\n')
    out.close()
    
    
def metric_function(xbags):
    metric = 0.0
    max_bags = 1000
    weight_limit = 50.0
    min_gifts = 3
    eval_loops = 100
    for xb in xbags:
        bg_w = 0.0
        for xe in range(eval_loops):
            bg = sum([we[xg.split('_')[0]]() for xg in xb])
            if len(xb) >= min_gifts and  bg < weight_limit:
                bg_w += bg
        metric += bg_w / eval_loops
    return metric

def objective_function(xbags, xgift_list):    
    #Add to bags
    for b in range(len(xbags)):
        for j in range(len(xgift_list[:10])): 
            xgift_list = shuffle(xgift_list, random_state=16)
            temp = copy.deepcopy(xbags[b])
            temp.append(str(xgift_list[j]))
            if metric_function([temp]) > metric_function([xbags[b]]):
                xgift_list.pop(j)
                xbags[b] = copy.deepcopy(temp)
                break
        if b % 100 == 0:
            print('Add to bags', b, metric_function(xbags))
    return xbags, xgift_list

we = {'horse': lambda: max(0, np.random.normal(5,2,1)[0]), 
     'ball': lambda: max(0, 1 + np.random.normal(1,0.3,1)[0]), 
     'bike': lambda: max(0, np.random.normal(20,10,1)[0]), 
     'train': lambda: max(0, np.random.normal(10,5,1)[0]), 
     'coal': lambda: 47 * np.random.beta(0.5,0.5,1)[0], 
     'book': lambda: np.random.chisquare(2,1)[0], 
     'doll': lambda: np.random.gamma(5,1,1)[0], 
     'blocks': lambda: np.random.triangular(5,10,20,1)[0], 
     'gloves': lambda: 3.0 + np.random.rand(1)[0]}

df = pd.read_csv('submission_fun0.csv')
gift_list = pd.read_csv('../input/gifts.csv')
gift_list = list(gift_list['GiftId'].values)
bags = [[y for y in x.split(' ') if len(y)>0] for x in df['Gifts']]
gifts_in_bags = []
for i in range(len(bags)):
    gifts_in_bags += bags[i]
gift_list = [x for x in gift_list if x not in gifts_in_bags]

for i_ in range(2):
    print("Start..", len(gift_list), sum([len(x) for x in bags]), metric_function(bags))
    for i in range(1):
        print(i, '='*20)
        bags, gift_list = objective_function(bags, gift_list)
    print("Done..", len(gift_list), sum([len(x) for x in bags]), metric_function(bags))

for i in range(30):
    print(i, '='*20)
    bags, gift_list = shuffle_function(bags, gift_list)
    out = open('submission_fun' + str(i) + '.csv', 'w')
    out.write('Gifts\n')
    for b in bags:
        out.write(' '.join(b) + '\n')
    out.close()