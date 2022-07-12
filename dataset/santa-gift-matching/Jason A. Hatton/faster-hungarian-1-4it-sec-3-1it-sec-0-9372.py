
# coding: utf-8

#                ,'     '.                      /  `.
#                  /        ,\                     |    \
#  MERRY CHRISTMAS |;       ;;''-------.._         ;     \
#     from the     \;;     ,;' _.-'       `.     .-.\     ;
#       _  _ ___    ';;,,-''``''-._         \    '  \\    |
#  |\ |/ \|_) | |_|     /          `'-._     |    \  '    ;
#  | \|\_/| \ | | |     |               `.   |.-.  \     ;  _
#    _  _     _         \_;,.._           ',/ |  |  |   /,'` `\
#   |_)/ \|  |_      _,,_/` / .'-.          `,\  \  ;  ,'      |
#   |  \_/|_ |_   ,-'  ,--'` |   o`-.        ;;|  `-' ;  .-    |
#               ;`    /      '.__  O `'.     _/       | /`    /
#              /      |_,--._,'  `.   | `.;/'  `'-. _ ;.-._,.'
#              /    _,'      |    |'-'  / |   .--.,' '\o  )_
#            .-;   `-..._     `-.-'     | \      _| O .'-'` `.
#          /`  |         '-'7     \  |  |  |'--'` /'-'.--.    `.
#        .| | <;            \      :-\  |  | ___ /    '._'     ;
#       / '-\__)\            '.    ;__,/   ;`   | .-.          |
#    ,.--.,  /   `.            \  /        |'--.\  |           /'-.
#   /'     `;`     `,           '/        /'--.  `-`.         / _  \
#   \,,-.   ;.       '-.._              ,'     `;    `-.___.-'`/'   |
#   /    `-,'.'.     /     `'-.__   _,-'    .--'/             |     /
#  /    ,  \  '.` f\/     /      `''`     /`   /              `.__.'\
# |     ;  |.   `/ .`'-._;             __.L   ;                  \  ;
# \    ,'`-' `._/ / `'-. |._____....--'_,-7\ /                   |   \
#  `''` |;,     `-.`'-.| |__.....--'''`_,'`|;                    \_|  |
#        \;,       `'-.__|__.,,__,,.."'`   ;|                      ;`-'
#         `:_             `'`              / \                     /
#            `._                    ,.__.-'   `._             mx .'
#           .-,|`-...________.,---'`|_           `-....____..--'`
#  ._       |   `.     ,-|     _,,,-' `.             \   |  /
#    '-..   .-._, `''"/   `---''       |              `. | ;        _
#        _,'   \;`-._ \_      _,-'`-.-'               _| | |_   .--'
#     .'`           `'"`|"`"`-;/     `' -. -..__   ,-' ` |   `'. _
#..--'\_       _,-'\___'    ,             |     ' / ,'  ,'.  `._) '-..
#      ' `"'''`         '---'`-..._____.-'        '-'--'   `'-' 

###################################################################################
#  A significant amount of RAM is required for good performance here.
#  16 GB should do it.  
#  The NumPy array is significantly more performant than a dictionary
#  But you need RAM to hold such large arrays.
#  Kaggle Kernels are not very performant.  You'll only get 2.3 to 3.2 it/sec.
#  This is better than the original which averages around 1.3 it/sec for me on Kaggle
#  My PC is fairly old and went from 2.7 it/sec to 4.9 it/sec
###################################################################################
#  Credit mostly to https://www.kaggle.com/gaborfodor
#  This is a forked kernel from 
#  https://www.kaggle.com/gaborfodor/improve-with-the-hungarian-method-0-9342
###################################################################################
#  You should have time for 15 loops before Kaggle kills the kernel.
#  Once this happens, simply feed the kernel output back as INPUT_SUBMISSION
###################################################################################


import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import datetime as dt
from tqdm import tqdm
from numba import jit


N_CHILDREN = 1000000
N_GIFT_TYPE = 1000
N_GIFT_QUANTITY = 1000
N_GIFT_PREF = 1000
N_CHILD_PREF = 10
TWINS = int(0.004 * N_CHILDREN)

CHILD_PREF = pd.read_csv('../input/santa-gift-matching/child_wishlist.csv', header=None).drop(0, 1).values
GIFT_PREF = pd.read_csv('../input/santa-gift-matching/gift_goodkids.csv', header=None).drop(0, 1).values


GIFT_HAPPINESS = (1. / (2 * N_GIFT_PREF)) * np.ones(shape=(N_GIFT_TYPE,N_CHILDREN),dtype=np.float32)
for g in range(N_GIFT_TYPE):
    for i, c in enumerate(GIFT_PREF[g]):
        GIFT_HAPPINESS[g,c] = -1. * (N_GIFT_PREF - i) / N_GIFT_PREF

CHILD_HAPPINESS = (1. / (2 * N_CHILD_PREF)) * np.ones(shape=(N_CHILDREN,N_GIFT_TYPE),dtype=np.float32)
for c in range(N_CHILDREN):
    for i, g in enumerate(CHILD_PREF[c]):
        CHILD_HAPPINESS[c,g] = -1. * (N_CHILD_PREF - i) / N_CHILD_PREF

GIFT_IDS = np.array([[g] * N_GIFT_QUANTITY for g in range(N_GIFT_TYPE)]).flatten()

@jit(nopython=False)
def my_avg_normalized_happiness(pred):
    total_child_happiness = 0
    total_gift_happiness = np.zeros(1000)
    for c, g in pred:
        total_child_happiness +=  -CHILD_HAPPINESS[c,g]
        total_gift_happiness[g] += -GIFT_HAPPINESS[g,c]
    nch = total_child_happiness / N_CHILDREN
    ngh = np.mean(total_gift_happiness) / 1000
    print('normalized child happiness', nch)
    print('normalized gift happiness', ngh)
    return nch + ngh

@jit(nopython=False)
def optimize_block(child_block, current_gift_ids):
    gift_block = current_gift_ids[child_block]
    C = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
    for i in range(BLOCK_SIZE):
        c = child_block[i]
        for j in range(BLOCK_SIZE):
            g = GIFT_IDS[gift_block[j]]
            C[i, j] = CHILD_HAPPINESS[c,g] + GIFT_HAPPINESS[g,c]
    row_ind, col_ind = linear_sum_assignment(C)
    return (child_block[row_ind], gift_block[col_ind])


BLOCK_SIZE = 400
INITIAL_SUBMISSION = '../input/nothing-fancy-just-some-heuristics-0-9372/heuristicSub.csv'
N_BLOCKS = (N_CHILDREN - TWINS) / BLOCK_SIZE
print('Block size: {}, n_blocks {}'.format(BLOCK_SIZE, N_BLOCKS))

subm = pd.read_csv(INITIAL_SUBMISSION)
initial_anh = my_avg_normalized_happiness(subm[['ChildId', 'GiftId']].values.tolist())
print(initial_anh)
subm['gift_rank'] = subm.groupby('GiftId').rank() - 1
subm['gift_id'] = subm['GiftId'] * 1000 + subm['gift_rank']
subm['gift_id'] = subm['gift_id'].astype(np.int64)
current_gift_ids = subm['gift_id'].values

start_time = dt.datetime.now()
for i in range(15):
    child_blocks = np.split(np.random.permutation(range(TWINS, N_CHILDREN)), N_BLOCKS)
    for child_block in tqdm(child_blocks[:500]):
        cids, gids = optimize_block(child_block, current_gift_ids=current_gift_ids)
        current_gift_ids[cids] = gids
    subm['GiftId'] = GIFT_IDS[current_gift_ids]
    anh = my_avg_normalized_happiness(subm[['ChildId', 'GiftId']].values.tolist())
    end_time = dt.datetime.now()
    print(i, anh, (end_time-start_time).total_seconds())
    subm[['ChildId', 'GiftId']].to_csv('./submission.csv', index=False)
