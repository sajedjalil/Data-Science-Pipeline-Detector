import numpy as np
import pandas as pd
import time
import math
import numba
import sympy
from sympy.utilities.iterables import multiset_permutations

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

def gen_subprobs(n, k):
    return np.array([[i, i + k] for i in range(n - k)], dtype=np.int32)

def gen_perms(k):
    idx = np.arange(1, k - 1, dtype=np.int32)
    perm = []
    for p in multiset_permutations(idx):
        perm.append(np.hstack([[0], p, [k-1]])) # head + perm + tail
    return np.array(perm, dtype=np.int32)

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

@numba.jit('i4(f4[:], i4[:,:,:], i8, i4[:,:], i4[:], f4[:,:], i4[:], i4[:,:])', nopython=True, parallel=True)
def finetune_(best_improvement, best_perm, k, subprobs, tour, loc, is_prime, perm):
    updated = 0
    init_tour = np.arange(k).astype(np.int32)
    # parallel loop
    for t in numba.prange(subprobs.shape[0]):
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
        costs = np.empty((perm.shape[0],), dtype=np.float32)
        init_cost = subprob_cost(init_tour, s_dist_mat, s_10th, s_prime)
        for i in range(perm.shape[0]):
            costs[i] = subprob_cost(perm[i], s_dist_mat, s_10th, s_prime)
        best_index = np.argmin(costs)
        delta = init_cost - costs[best_index]
        if delta > best_improvement[t]:
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
tour = load_tour("../input/winter-avalanche-2/submission.csv")
init_cost = cost_santa2018(tour, loc, is_prime)
print("initial cost", init_cost)

# subprob size + fixed head + fixed tail. 5!=120, 7!=5040(26sec), 9!=362880(1949sec)
K = 9 + 2
subprobs = gen_subprobs(city_id.shape[0], K)
subprob_perms = gen_perms(K)
print("** perm={}!".format(K-2))
t = time.time()
updated = finetune(K, subprobs, tour, loc, is_prime, subprob_perms)
cost = cost_santa2018(tour, loc, is_prime)
print("cost: {:.4f}, improvement: {:.4f}, found: {}, time: {:.2f}".format(cost, init_cost - cost, updated, time.time() - t))
save_tour("submission.csv", tour)