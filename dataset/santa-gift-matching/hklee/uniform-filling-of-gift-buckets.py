### Happy XMAS for Christians, Happy New Year for Christians and Christians. from hklee

rudolf_lyrics = """
You know Dasher, and Dancer, and Prancer, and Vixen
Comet, and Cupid, and Donner and Blitzen
But do you recall the most famous reindeer of all

Rudolph the red-nosed reindeer had a very shiny nose
And if you ever saw it, you would even say it glows
All of the other reindeer used to laugh and call him names
They never let poor Rudolph join in any reindeer games

Then one foggy Christmas Eve Santa came to say
"Rudolph with your nose so bright, won't you guide my sleigh tonight?"
Then how the reindeer loved him as they shouted out with glee
"Rudolph the red-nosed reindeer, you'll go down in history"

Rudolph the red-nosed reindeer had a very shiny nose
And if you ever saw it, you would even say it glows
All of the other reindeer used to laugh and call him names
They never let poor Rudolph join in any reindeer games

Then one foggy Christmas Eve Santa came to say
"Rudolph with your nose so bright, won't you guide my sleigh tonight?"
Then how the reindeer loved him as they shouted out with glee
"Rudolph the red-nosed reindeer, you'll go down in history"
""".replace('\n', '*')


import numpy as np, pandas as pd

print('load data')
child_prefs = pd.read_csv('../input/child_wishlist.csv', header=None).values[:,1:]
gift_prefs = pd.read_csv('../input/gift_goodkids.csv', header=None).values[:,1:]

n_children = 1000000 # n children to give
n_gift_type = 1000 # n types of gifts available
n_gift_quantity = 1000 # each type of gifts are limited to this quantity
n_child_pref = 10 # number of gifts a child ranks
n_gift_pref = 1000 # number of children a gift ranks
twins = 4000
ratio_gift_happiness = 2
ratio_child_happiness = 2

print('combine all nontrivial child-gift pairs and assign happiness on them.')
df_santa = pd.DataFrame({
    'gift':np.repeat(np.arange(n_gift_type), n_gift_pref) ,
    'child':gift_prefs.flatten(),
    'santa_happ':np.tile(np.linspace(n_gift_pref, 1, n_gift_pref) * 2, (n_gift_type, 1)).flatten()})

df_child = pd.DataFrame({
    'child':np.repeat(np.arange(n_children), n_child_pref),
    'gift': child_prefs.flatten(),
    'child_happ': np.tile(np.linspace(n_child_pref, 1, n_child_pref) * 2, (n_children, 1)).flatten()})

df = df_child.merge(df_santa, on=['child', 'gift'], how='outer')

df['child_happ'] = df.child_happ.fillna(-1)
df['santa_happ'] = df.santa_happ.fillna(-1)

max_child_happ = 2 * n_child_pref
max_gift_happ  = 2 * n_gift_pref

df['happiness']  = df.child_happ / max_child_happ + df.santa_happ / max_gift_happ    

print('split singles and twins, and take care of twins')
df_twins = df[df.child < twins].reset_index(drop=True)
df_twins['child_t'] = df_twins.child.apply(lambda x: x + 1 if x//2*2==x else x - 1)  # other me

# add happiness of the other's choice
df_tmp = df_twins[['child_t', 'gift']].merge(df_twins[['child', 'gift', 'child_happ']], left_on=['child_t', 'gift'], right_on=['child', 'gift'], how='left')
df_twins['child_happ_t'] = df_tmp.child_happ.fillna(-1)

df_tmp = df_twins[['child_t', 'gift']].merge(df_santa, left_on=['child_t', 'gift'], right_on=['child', 'gift'], how='left')
df_twins['santa_happ_t'] = df_tmp.santa_happ.fillna(-1)

df_twins['happiness'] = (df_twins.child_happ + df_twins.child_happ_t) / max_child_happ + (df_twins.santa_happ + df_twins.santa_happ_t) / max_gift_happ 

# remove other me 
ii = df_twins.child > df_twins.child_t
df_twins.loc[ii, 'child'] = df_twins.child_t[ii]

df_twins = df_twins[['child', 'gift', 'happiness']].copy()
df_twins['n'] = 2

print('merge singles and twins')
df1 = df[['child', 'gift', 'happiness']][df.child >= twins].reset_index(drop=True)
df1['n'] = 1

df_dist = pd.concat([df_twins, df1], ignore_index=True)
df_dist.sort_values(by='happiness', ascending=False, inplace=True)
df_dist.reset_index(drop=True, inplace=True)

print('create lists of candidate children and their happiness on each gift. it takes some secs.')
candidates = []  # for gifts
chappiness = []
for gift in range(n_gift_type):
    a = df_dist[df_dist.gift==gift][['child', 'happiness']]
    candidates.append(a.child.values)
    chappiness.append(a.happiness.values)
    
clen = np.array([len(x) for x in candidates])
crt = np.zeros_like(clen)    

print('distribute gifts by filling gift list uniformly.')
preds = np.ones(n_children, dtype=np.int32) * -1
lefts = np.ones(n_gift_type, dtype=np.int32) * 1000

total_happ = 0

lefts_blocked = np.zeros_like(lefts)

for i in range(n_children):
    
    gifts = np.argsort(lefts)[::-1]
    
    maxlefts = lefts[gifts[0]]
    if maxlefts == 0: break
    
    maxhapp = -1000
    maxchild = -1
    maxgift  = -1
    
    for gift in gifts:
        if lefts[gift] != maxlefts:
            break
            
        candid = candidates[gift]
        chapp  = chappiness[gift]
        k = crt[gift]

        while (k < clen[gift]):
            child = candid[k]
            if preds[child] == -1: # not assigned yet
                happ  = chapp[k]
                if (happ > maxhapp) and ( (child >= twins) or (lefts[gift] >= 2)):
                    maxhapp  = happ
                    maxchild = child
                    maxgift  = gift
                break
            else:
                k = k + 1
        crt[gift] = k
        
    if maxchild == -1: 
        for gift in gifts:
            if lefts[gift] != maxlefts:
                break
            lefts_blocked[gift] = lefts[gift]
            lefts[gift] = 0
    else:
        lefts[maxgift] -= 1
        preds[maxchild] = maxgift
        if maxchild < twins:
            lefts[maxgift] -= 1
            preds[maxchild+1] = maxgift
        crt[maxgift] += 1
        total_happ += maxhapp
        
    if i % 500 == 0:
        print("{}/{}, happiness={:.4f} {}".format(i, n_children, total_happ / 1000000, rudolf_lyrics[(i//500)%len(rudolf_lyrics)]))
        
#expected score
#expected_happiness = (total_happ + np.sum(lefts_blocked) * (-1/max_child_happ + -1/max_gift_happ)) / ((n_children + n_gift_type * n_gift_quantity)/2)
expected_happiness = (total_happ + np.sum(lefts_blocked) * (-1/max_child_happ + -1/max_gift_happ)) / 1000000
print('expected happiness={:.4f}'.format(expected_happiness))

# give remaining gifts to unfortunate kids.
lefts = lefts_blocked

gifts = []
for gift in np.where(lefts >= 2)[0]:
    n = lefts[gift] // 2 * 2  # twins comes first
    lefts[gift] -= n
    gifts = gifts + [ gift ] * n

for gift in np.where(lefts ==1)[0]:
    gifts.append(gift)
    
preds[preds == -1] = np.array(gifts)

# submit
pd.DataFrame({'ChildId':np.arange(n_children), 'GiftId':preds}).set_index('ChildId').to_csv('uniform_fill.csv')

print('done.')