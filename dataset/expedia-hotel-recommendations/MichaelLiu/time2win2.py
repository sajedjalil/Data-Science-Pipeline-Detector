# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import random

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

p_train = '../input/train.csv'
p_test = '../input/test.csv'
p_out = 'pred.csv'

def get_key(row):
    s = row['site_name']
    uc = row['user_location_country']
    d = 5
    try:
        x = float(row['orig_destination_distance'])
        if x < 10: d = 1
        elif x < 100: d = 2
        elif x < 1000: d = 3
        elif x >= 1000: d = 4
    except:
        d = 5
    m = row['is_mobile']
    p = row['is_package']
    if row['srch_adults_cnt'] == 1: a = 1
    elif row['srch_adults_cnt'] == 2: a = 2
    else: a = 3
    if row['srch_children_cnt'] == 0: ch = 0
    elif row['srch_children_cnt'] == 1: ch = 1
    else: ch = 2
    t = row['srch_destination_type_id']
    hc = row['hotel_country']
    did1 = '\1'.join([str(v) for v in [s, uc, d, m, p, a, ch, t, hc]])
    did2 = '\1'.join([str(v) for v in [ch, t, hc]])
    return did1, did2

c_stat = {}
d_stat = {}
d2_stat = {}
nrows = 0
for row in csv.DictReader(open(p_train)):
    if random.randint(0,5) != 1: continue
    did, did2 = get_key(row)
    cid = row['hotel_cluster']
    if did not in d_stat: 
        d_stat[did] = {}
    d_stat[did][cid] = d_stat[did].get(cid, 0) + 1
    if did2 not in d2_stat: 
        d2_stat[did2] = {}
    d2_stat[did2][cid] = d2_stat[did2].get(cid, 0) + 1
    c_stat[cid] = c_stat.get(cid, 0) + 1
    #nrows += 1
    #if nrows > 100000: 
    #    break

print(c_stat)
#print(d_stat)

def get_topk(d_stat, k=5, thres=0.005):
    d_hot = {}
    for did in d_stat:
        d_tot = sum(d_stat[did].values())
        if d_tot < k:
            continue
        for cid in d_stat[did]:
            d_stat[did][cid] /= float(d_tot)
        sort_cid = sorted(d_stat[did].items(), key=lambda d:-d[1])
        ret = ' '.join([str(v[0]) for v in sort_cid if v[1]>thres][:k])
        d_hot[did] = ret
    return d_hot

c_hot = ' '.join([str(v[0]) for v in sorted(c_stat.items(), key=lambda d:-d[1])][:3])
d_hot = get_topk(d_stat, 5, 0.005)
d2_hot = get_topk(d2_stat, 5, 0.005)

#print(d_hot)
print(c_hot)

with open(p_out, 'w') as fo:
    fo.write('id,hotel_cluster\n')
    nrows = 0
    for row in csv.DictReader(open(p_test)):
        eid = row['id']
        did, did2 = get_key(row)
        if did in d_hot:
            ret = d_hot[did]
        elif did2 in d2_hot:
            ret = d2_hot[did2]
        else:
            ret = c_hot
        fo.write('%s,%s\n' % (eid, ret))
        #nrows += 1
        #if nrows > 100:
        #    break
        
        