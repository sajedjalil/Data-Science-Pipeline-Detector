# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import itertools

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
def generate_sample():
    sample = {}
    sample['ball'] = [max(0, 1 + np.random.normal(1,0.3,1)[0]) for i in range(11000)]
    sample['bike'] = [max(0, np.random.normal(20,10,1)[0]) for i in range(5000)]
    sample['blocks'] = [np.random.triangular(5,10,20,1)[0] for i in range(10000)]
    sample['book'] = [np.random.chisquare(2,1)[0] for i in range(12000)]
    sample['coal'] = [47 * np.random.beta(0.5,0.5,1)[0] for i in range(1660)]
    sample['doll'] = [np.random.gamma(5,1,1)[0] for i in range(10000)]
    sample['gloves'] = [3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0] for i in range(2000)]
    sample['horse'] = [max(0, np.random.normal(5,2,1)[0]) for i in range(10000)]
    sample['train'] = [max(0, np.random.normal(10,5,1)[0]) for i in range(10000)]
    return sample

lst_all = ['horse', 'ball','train', 'book', 'doll', 'blocks', 'gloves', 'coal', 'bike']
sample = generate_sample()
percentile25 = {}
percentile75 = {}
for cate in lst_all:
    percentile25[cate] = np.percentile(sample[cate], 25)
    percentile75[cate] = np.percentile(sample[cate], 75)

dispatch = {
    "horse": lambda:max(0, np.random.normal(5,2,1)[0]),
    "ball": lambda:max(0, 1 + np.random.normal(1,0.3,1)[0]),
    "bike": lambda:max(0, np.random.normal(20,10,1)[0]),
    "train": lambda:max(0, np.random.normal(10,5,1)[0]),
    "coal": lambda:47 * np.random.beta(0.5,0.5,1)[0],
    "book": lambda:np.random.chisquare(2,1)[0],
    "doll": lambda:np.random.gamma(5,1,1)[0],
    "blocks": lambda:np.random.triangular(5,10,20,1)[0],
    "gloves": lambda:3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]
    }

def init_count():
    count = {
        "horse": 1000,
        "ball": 1100,
        "bike": 500,
        "train": 1000,
        "coal": 166,
        "book": 1200,
        "doll": 1000,
        "blocks": 1000,
        "gloves": 200
    }
    return count
    
def init_limit():
    limit = {
        "horse": 6,
        "ball": 8,
        "bike": 2,
        "train": 3,
        "coal": 0,
        "book": 4,
        "doll": 6,
        "blocks": 3,
        "gloves": 4
    }
    return limit

def combination_generator(a, n, k):
    if n > k: yield None
    elif n ==0: yield []
    elif n == 1: yield a*k
    elif n == 2: 
        for i in range(k-1): yield [a[0]]*(1+i) + [a[1]]*(k-i-1)
    elif k == 3: yield [a[0], a[1], a[2]]
    else:
        generator = itertools.chain(
            combination_generator(a,n,k-2), 
            combination_generator(a[1:],n-1,k-2), 
            combination_generator(a[:-1],n-1,k-2), 
            combination_generator(a[1:-1],n-2,k-2) )
        for iter in generator: 
            if iter is not None: yield [a[0]] + iter + [a[-1]]

def simulator(lst, n):
    s = 0
    for i in range(10000):
        total_w = 0
        for i in range(n):
            w = 0
            for x in lst:
                w += dispatch[x]()
            if w <= 50:
                total_w += w
        s += total_w
    return s*1.0/10000

def simulator_wrapper(n, lst_candidate, limit, thres1=55, thres2=29):
    max_w = -1
    best_comb = None
    cnt = 0
    for i in range(1,n+1):
        for lst in itertools.combinations(lst_candidate, i):
            for iter in combination_generator(list(lst), i, n):
                w1 = sum([percentile25[cate] for cate in iter])
                w2 = sum([percentile75[cate] for cate in iter])
                flag = True
                c = {}
                for x in iter:
                    c[x] = c.get(x, 0) + 1
                for x in iter:
                    if c[x] > limit[x]:
                        flag = False
                        break
                if flag and w2 <= thres1 and w1 > thres2:
                    cnt += 1
                    w = simulator(iter,1)
                    if w > max_w:
                        max_w = w
                        best_comb = iter
    return cnt, max_w, best_comb

f_out = open('submission.csv', 'w')
f_out.write('Gifts\n')
count = init_count()
shuffle_map = {}
for k,v in count.items():
    shuffle_map[k] = np.random.permutation(np.arange(v))
limit = init_limit()
n_bags = 0
w_bags = 0
thres1 = {9:50, 8:50, 7:50, 6:50, 5:50, 4:100, 3:100, 2:100}
thres2 = {9:30, 8:25, 7:20, 6:20, 5:15, 4:10, 3:5, 2:5}

while n_bags < 1000:
    lst_candidate = [x for x in lst_all if count[x] > 0]
    print('Searching from',lst_candidate)
    max_w = -1
    best_comb = None
    for i in range(3,10):
        _, w, comb = simulator_wrapper(i, lst_candidate, limit, thres1[len(lst_candidate)], thres2[len(lst_candidate)])
        if w > max_w:
            max_w = w
            best_comb = comb
            print('\t', w, comb)
    if best_comb is None:
        print('None')
        break
    
    print('Packaging...')
    while n_bags < 1000:
        bag = []
        tmp_count = {}
        flag = True
        for x in best_comb:
            if count[x] - tmp_count.get(x, 0) <= 0:
                flag = False
                break
            else:
                tmp_count[x] = tmp_count.get(x, 0) + 1
                idx = count[x] - tmp_count[x]
                bag.append('%s_%s' % (x, shuffle_map[x][idx]))
                #bag.append('%s_%s' % (x, idx))
        if flag:
            for x, c in tmp_count.items():
                count[x] = count[x] - c
            bag = ' '.join(bag) + '\n'
            f_out.write(bag)
            n_bags += 1
            w_bags += max_w
        else:
            break
    for x, c in count.items():
        limit[x] = min(count[x], limit[x])
    print('\tn_bags=%s, w_bags=%s' % (n_bags, w_bags))

f_out.close()
print('Done')