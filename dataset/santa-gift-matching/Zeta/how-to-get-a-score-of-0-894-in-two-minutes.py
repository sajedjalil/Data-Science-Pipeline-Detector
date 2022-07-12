import time
import pandas as pd
import numpy as np
import math
from tqdm import tqdm

INPUT_PATH = '../input/'
__KagglerID__ = 'Zeta'
__KaggleProfile__ = 'https://www.kaggle.com/zeta2191622'
__KernelFor__ = 'SantaGift'

def lcm(a, b):
    lcm_ = a * b // math.gcd(a, b)
    return lcm_, lcm_/a, lcm_/b
def get_indices_pandas(data):
    d_ = data.ravel()
    f = lambda x: np.unravel_index(x.index, data.shape)
    return pd.Series(d_).groupby(d_).apply(f)
def get_indices_pandas2(data):
    d_ = data.ravel()
    f = lambda x: np.unravel_index(x.index, data.shape)[0][0]
    return pd.Series(d_).groupby(d_).apply(f)
    
def avgNH(pred):
    """ Adapted from TeraFlops's code: https://www.kaggle.com/sekrier/50ms-scoring-just-with-numpy"""
    chigif = np.full((1000000, 1000), -1,dtype=np.int16)
    VAL = (np.arange(200,0,-2)+1)
    for c in tqdm(range(1000000)):
        chigif[c, wish[c]] += VAL
        
    gifchi = np.full((1000, 1000000), -1,dtype=np.int16)
    VAL = (np.arange(2000,0,-2)+1)
    for g in tqdm(range(1000)):
        gifchi[g, gift[g]] += VAL
        
    TCH = np.sum(chigif[range(n_children),pred])
    TSH = np.sum(gifchi[pred,range(n_children)])
       
    ret = float(math.pow(TCH*multiplier1,3) + \
            math.pow(np.sum(TSH)*multiplier2,3)) / float(math.pow(common_denom,3))
    print('Score: %.8f'%ret)
#%%
TimeOn = time.time()
wish = pd.read_csv(INPUT_PATH + 'child_wishlist_v2.csv', header = None).as_matrix()[:, 1:]
gift = pd.read_csv(INPUT_PATH + 'gift_goodkids_v2.csv', header = None).as_matrix()[:, 1:]
n_children = wish.shape[0] # n children to give
n_gift_type = gift.shape[0] # n types of gifts available
n_gift_quantity = gift.shape[1] # each type of gifts are limited to this quantity
n_gift_pref = wish.shape[1] # number of gifts a child ranks
n_child_pref = gift.shape[1] # number of children a gift ranks
n_twins = math.ceil(0.04 * n_children / 2.) 
twins = int(n_twins * 2)    # 4% of all population, rounded to the closest number
n_triplets = math.ceil(0.005 * n_children / 3.) 
triplets = int(n_triplets  * 3)   # 0.5% of all population, rounded to the closest number
n_single_child = n_children-twins-triplets
ratio_gift_happiness = 2
ratio_child_happiness = 2
max_child_happiness = n_gift_pref * ratio_child_happiness
max_gift_happiness = n_child_pref * ratio_gift_happiness
denominator1 = n_children*max_child_happiness
denominator2 = n_gift_quantity*max_gift_happiness*n_gift_type
common_denom,multiplier1, multiplier2 = lcm(denominator1, denominator2) 
print('%d gift for %d children. Time: %.2f seconds'%(n_children, n_children, time.time()-TimeOn))
#%% for each gift, collect all kids' ID who rank this gift 
GiftRank = get_indices_pandas(wish) ## return sereis of tubple (giftId==row,giftrank==column))
wish_unravel = np.ravel(wish)
RG_count = pd.Series(wish_unravel).groupby(wish_unravel).count()
loopSeq = RG_count.argsort() # sort gift by number of children ranking it
avgNCH = -1*np.ones((n_gift_type, n_gift_quantity), dtype = np.int64) # 1000 by 1000 matrix storing child ID 
exKids = set([])
gift_count = np.zeros((n_gift_type, ), dtype=np.int16) ## counter for each gift
for gidx in loopSeq: ## assgining the least favarable gift first!
    rChildIdx = GiftRank[gidx][0]  ## which children ranked this gift
    rgiftRank = GiftRank[gidx][1]  ## How does each child rank this gift
    nkrnk = np.argsort(rgiftRank)  ## sorted child ID by their ranks
    assignIdx = gift_count[gidx]
    for idx in nkrnk:
        newkid=rChildIdx[idx] 
        if assignIdx < n_gift_quantity:
            if not (newkid in exKids):
                if newkid < triplets:
                    T0 = newkid-np.mod(newkid, 3)
                    if (not (T0 in exKids)) and assignIdx < n_gift_quantity-3:
                        exKids.update([T0, T0+1, T0+2])
                        avgNCH[gidx][assignIdx] = T0
                        avgNCH[gidx][assignIdx+1] = T0+1
                        avgNCH[gidx][assignIdx+2] = T0+2
                        assignIdx += 3

                elif newkid >= triplets and newkid < triplets + twins:
                    T0 = newkid-np.mod(newkid-1, 2)
                    if (not (T0 in exKids)) and assignIdx < n_gift_quantity-2:
                        exKids.update([T0, T0+1])
                        avgNCH[gidx][assignIdx] = T0
                        avgNCH[gidx][assignIdx+1] = T0+1
                        assignIdx += 2                 
                else:
                    exKids.update([newkid])
                    avgNCH[gidx][assignIdx] = newkid
                    assignIdx += 1                    
        else:
            break ## no more than 1000 gift per type
    gift_count[gidx] = assignIdx   
print('Gift unassigned: %d. Time: %.2f seconds'%(n_children-np.sum(gift_count), time.time()-TimeOn))
#%%
AllIDs = set(range(n_children))
AllIDs.difference_update(exKids)  ## kids without gift
outTup = get_indices_pandas(avgNCH) 
Gift_left = outTup[-1]  ## gift withou kids
leftGiftIds = np.unique(Gift_left[0])
unhappyTrip0 = [x for x in np.arange(0, triplets, 3) if x in AllIDs] # tripplets without gift
unhappyTwin0 = [x for x in np.arange(triplets, triplets+twins, 2) if x in AllIDs] # twins without gift
# triplets withou gift
for idx in unhappyTrip0:
    for gidx in leftGiftIds:
        assignIdx = gift_count[gidx]
        if assignIdx < n_gift_quantity-3:
            exKids.update([idx, idx+1, idx+2])
            avgNCH[gidx][assignIdx] = idx
            avgNCH[gidx][assignIdx+1] = idx+1
            avgNCH[gidx][assignIdx+2] = idx+2
            assignIdx += 3
            gift_count[gidx] = assignIdx
            break       
# twins without gift           
for idx in unhappyTwin0:
    for gidx in leftGiftIds:
        assignIdx = gift_count[gidx]
        if assignIdx < n_gift_quantity-2:
            exKids.update([idx, idx+1])
            avgNCH[gidx][assignIdx] = idx
            avgNCH[gidx][assignIdx+1] = idx+1
            assignIdx += 2
            gift_count[gidx] = assignIdx
            break

AllIDs = set(range(n_children))
AllIDs.difference_update(exKids)  ## kids without gift
# single kid without gift
for idx, childID in enumerate(AllIDs):
    for gidx in leftGiftIds:
        assignIdx = gift_count[gidx]
        if assignIdx < n_gift_quantity:
            if avgNCH[gidx][assignIdx] == -1:
                exKids.update([childID])
                avgNCH[gidx][assignIdx] = childID
                assignIdx += 1
                gift_count[gidx] += 1   
                break
            else:
                print('something wrong')
                break
print('Gift unassigned: %d. Time: %.2f seconds'%(n_children-np.sum(gift_count), time.time()-TimeOn))
#%% sanity check
AllIDs = set(range(n_children))
AllIDs.difference_update(exKids) 
assert len(AllIDs) == 0 # all gift is out!
assert np.sum(gift_count) == n_children # total gift sent
assert np.unique(gift_count) == n_gift_quantity # number of each gift type sent
assert len(np.unique(avgNCH)) == n_children # each child should have a gift
#%%
print('Start scoring. Time: %.2f'%(time.time()-TimeOn))
outGift = get_indices_pandas2(avgNCH) 
score = avgNH(outGift)
Outsfx = __KagglerID__+__KernelFor__
out = open('{}_Score_{}.csv'.format(Outsfx, score),  'w')
out.write('ChildId,GiftId\n')
for i in range(len(outGift)):
    out.write(str(i) + ',' + str(outGift[i]) + '\n')
out.close()
print('All done. Time: %.2f seconds'%(time.time()-TimeOn))