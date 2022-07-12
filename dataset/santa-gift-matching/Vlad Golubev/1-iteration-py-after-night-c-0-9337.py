import numpy as np
import pandas as pd

wish = pd.read_csv('../input/santa-gift-matching/child_wishlist.csv', header=None).as_matrix()[:, 1:]
gift = pd.read_csv('../input/santa-gift-matching/gift_goodkids.csv', header=None).as_matrix()[:, 1:]
df = pd.read_csv('../input/c-submission/cpp_sub.csv') #all night optimization on C++

preds = df['GiftId'].values
scores = np.zeros((len(gift), len(wish)), dtype='int16') #int16 for minimize used memory
scores.fill(-101)
w2g = [[] for i in range(len(gift))]

for i in range(len(gift)):
    for j in range(len(gift)):
        wid = gift[i][j]
        scores[i][wid] += (len(gift)-j)*2+1; #for minimize used memory
        w2g[i].append(wid)

for i in range(len(wish)):
    for j in range(10):
        gid = wish[i][j]
        if scores[gid][i] == -101:
            w2g[gid].append(i)
        scores[gid][i] += (10-j)*200+100 #for minimize used memory

def optimization(preds, scores, w2g):
    for i in range(4000, len(wish)):
        gid1 = preds[i]
        for j in w2g[gid1]:
            if j < 4000:
                continue
            gid2 = preds[j]
            t1 = scores[gid1][i]+scores[gid2][j]
            t2 = scores[gid2][i]+scores[gid1][j]
            if t2 > t1:
                preds[i] = gid2;
                preds[j] = gid1;
                return #break - if wanna more
            
optimization(preds, scores, w2g)
df['GiftId'] = preds
df.to_csv('opt_sub.csv', index=False)