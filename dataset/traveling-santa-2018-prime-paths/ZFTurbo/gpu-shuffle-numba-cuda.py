import os
# set env for numba.cuda
os.environ['NUMBAPRO_NVVM']='/usr/local/cuda/nvvm/lib64/libnvvm.so' 
os.environ['NUMBAPRO_LIBDEVICE']='/usr/local/cuda/nvvm/libdevice/'

import numpy as np
import pandas as pd
import time
import math
import numba
from numba import cuda
import sympy
from sympy.utilities.iterables import multiset_permutations
from tqdm import tqdm

# load
def load_cities(filename):
    cities = pd.read_csv(filename)
    city_id = cities.CityId.astype(np.int32)
    loc = np.vstack([cities.X.astype(np.float32), cities.Y.astype(np.float32)]).transpose()
    is_prime = np.array([1 if sympy.isprime(i) else 0 for i in city_id], dtype=np.int32)
    return (city_id, loc, is_prime)

def load_tour(filename):
    tour = pd.read_csv(filename)
    tour = tour.Path.values.astype(np.int32)
    return tour

# save
def save_tour(filename, tour):
    with open(filename, "w") as f:
        f.write("Path\n")
        for i in tour:
            f.write(str(i))
            f.write("\n")

# cost function for santa 2018

@numba.jit('f4(f4[:], f4[:])', nopython=True)
def euc_2d(a, b):
    xd = a[0] - b[0]
    yd = a[1] - b[1]
    return math.sqrt(xd * xd + yd * yd)

@numba.jit('f8(i4[:], f4[:,:], i4[:])', nopython=True, parallel=True)
def cost_santa2018(tour, loc, is_prime):
    dist = 0.0
    for i in numba.prange(1, tour.shape[0]):
        a = tour[i - 1]
        b = tour[i]
        d = euc_2d(loc[a], loc[b])
        if i % 10 == 0 and is_prime[a] == 0:
            d *= 1.1
        dist += d
    return dist

# finetune

# finetune
def gen_subprobs(n, k):
    return np.array([[i, i + k] for i in range(n - k + 1)], dtype=np.int32)

@numba.jit
def gen_perms(k):
    idx = np.arange(1, k - 1, dtype=np.int32)
    perm_pad = np.empty((math.factorial(k - 2), k), dtype=np.int32)
    perm_pad[:, 0] = 0
    perm_pad[:, 1:-1] = list(multiset_permutations(idx))
    perm_pad[:, -1] = k - 1
    return perm_pad

@numba.jit('f4(i4[:], f4[:,:], i4[:], i4[:])', nopython=True)
def subprob_cost(tour, dist_mat, s_10th, s_prime):
    dist = 0.0
    for t in range(1, tour.shape[0]):
        i = tour[t - 1]
        j = tour[t]
        d = dist_mat[i][j]
        if s_10th[t] != 0 and s_prime[i] != 0:
            d *= 1.1
        dist += d
    return dist
    
@cuda.jit
def subprob_cost_gpu(min_index, min_value, tours, s_dist_mat, s_10th, s_prime):
    index = cuda.grid(1)
    dist = 0.0
    if index < tours.shape[0]:
        tour = tours[index]
        for t in range(1, tour.shape[0]):
            i = tour[t - 1]
            j = tour[t]
            dist += s_dist_mat[i][j] + (s_10th[t] * s_prime[i] * 0.1 * s_dist_mat[i][j])
        cuda.atomic.min(min_value, 0, dist)
    cuda.syncthreads()
    if abs(min_value[0] - dist) < 1.0e-5:
        min_index[0] = index

@numba.jit
def finetune_(best_improvement, best_perm, k, subprobs, tour, loc, is_prime, perm):
    updated = 0
    init_tour = np.arange(k).astype(np.int32)
    perm_gpu = cuda.to_device(perm)
    for t in tqdm(range(subprobs.shape[0]), ncols=80):
        idx = np.arange(subprobs[t][0], subprobs[t][1]).astype(np.int32)
        s_loc = loc[tour[idx]]
        s_prime = (1 - is_prime[tour[idx]]).astype(np.int32)
        s_10th = np.empty((k,), dtype=np.int32)
        s_dist_mat = np.empty((k, k), dtype=np.float32)
        for i in range(k):
            s_10th[i] = 1 if (i + idx[0]) % 10 == 0 else 0
        for i in range(k):
            for j in range(i + 1, k):
                s_dist_mat[i][j] = s_dist_mat[j][i] = euc_2d(s_loc[i], s_loc[j])

        # brute force
        init_cost = subprob_cost(init_tour, s_dist_mat, s_10th, s_prime)
        min_value = np.empty((1,), dtype=np.float32)
        min_index = np.empty((1,), dtype=np.int32)
        min_value[0] = 100000.0
        min_index[0] = 0
        threadsperblock = 512
        blockspergrid = (perm.shape[0] + (threadsperblock - 1)) // threadsperblock
        subprob_cost_gpu[blockspergrid, threadsperblock](min_index, min_value, perm_gpu, s_dist_mat, s_10th.astype(np.float32), s_prime.astype(np.float32))

        best_index = min_index[0]
        delta = init_cost - min_value[0]
        if delta > 1e-4:
            # found improvement
            best_perm[t][0] = idx
            best_perm[t][1] = idx[perm[best_index]]
            best_improvement[t] = delta
            updated += 1
    return updated

def finetune(k, subprobs, tour, loc, is_prime, perm):
    updated = 0
    # numba does not seem to have get_thread_id() or lock(), so allocate extra buffers for threads
    best_perm = np.empty((subprobs.shape[0], 2, k), dtype=np.int32)
    best_improvement = np.zeros((subprobs.shape[0],), dtype=np.float32)
    # finetune
    finetune_(best_improvement, best_perm, k, subprobs, tour, loc, is_prime, perm)
    # update tour
    used = []
    updates = np.argsort(best_improvement)[::-1]
    for i in updates:
        delta = best_improvement[i]
        if delta > 0:
            s = best_perm[i][0][0]
            e = best_perm[i][0][-1]
            if any([(r[0] < s and s < r[1]) or (r[0] < e and e < r[1]) for r in used]):
                continue
            used.append((s, e))
            tour[best_perm[i][0]] = tour[best_perm[i][1]]
            updated += 1
        else:
            break
    return updated

# run

city_id, loc, is_prime = load_cities("../input/traveling-santa-2018-prime-paths/cities.csv")
# load jazivxt's Winter Avalanche2 result (https://www.kaggle.com/jazivxt/winter-avalanche-2)
tour = load_tour("../input/frunk-optimization/final_frunkopt_1515558.3918.csv")
init_cost = cost_santa2018(tour, loc, is_prime)
print("initial cost", init_cost)

# subprob size + fixed head + fixed tail. 9!=362880, 11!=39916800, 11!/9!=110
K = 11 + 2
subprobs = gen_subprobs(city_id.shape[0], K)
subprob_perms = gen_perms(K)
print("** perm={}!".format(K-2))
t = time.time()
updated = finetune(K, subprobs, tour, loc, is_prime, subprob_perms)
cost = cost_santa2018(tour, loc, is_prime)
print("cost: {:.4f}, improvement: {:.4f}, found: {}, time: {:.2f}".format(cost, init_cost - cost, updated, time.time() - t))
save_tour("submission_{}.csv".format(cost), tour)