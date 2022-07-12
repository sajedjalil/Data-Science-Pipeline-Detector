# Santa Hill Climbing 2019

import numba
from numba import njit
from numba.typed import Dict
import numpy as np
from time import time
from os import path
import csv

# const

DBL_MAX = 1e+308

N_FAMILES = 5000
N_DAYS = 100
N_CHOICES = 10
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125

MAX_DIFF = 300
MAX_DIFF2 = MAX_DIFF * 2
MAX_FAMILY_PER_DAY = 200

DATA_DIR = "../input/santa-workshop-tour-2019/"


# LUT for cost function

@njit(fastmath=True)
def build_cost_lut(family_size, family_choice):
    pref_cost = np.empty((N_FAMILES, N_DAYS), dtype=np.float64)
    acc1_cost = np.empty((MAX_DIFF2,), dtype=np.float64)
    acc_cost = np.empty((MAX_DIFF2, MAX_DIFF2), dtype=np.float64)
    penalty = np.empty((MAX_DIFF2, ), dtype=np.float64)

    for i in range(N_FAMILES):
        # preference cost
        n = family_size[i]
        pref_cost[i][:] = 500 + 36 * n + 398 * n
        pref_cost[i][family_choice[i][0]] = 0
        pref_cost[i][family_choice[i][1]] = 50
        pref_cost[i][family_choice[i][2]] = 50 + 9 * n
        pref_cost[i][family_choice[i][3]] = 100 + 9 * n
        pref_cost[i][family_choice[i][4]] = 200 + 9 * n
        pref_cost[i][family_choice[i][5]] = 200 + 18 * n        
        pref_cost[i][family_choice[i][6]] = 300 + 18 * n
        pref_cost[i][family_choice[i][7]] = 300 + 36 * n
        pref_cost[i][family_choice[i][8]] = 400 + 36 * n
        pref_cost[i][family_choice[i][9]] = 500 + 36 * n + 199 * n

    for i in range(MAX_DIFF2):
        # accounting cost
        acc1_cost[i] = max(0, (i - 125.0) / 400.0 * i ** 0.5)
        for j in range(MAX_DIFF2):
            diff = abs(j - MAX_DIFF)
            acc_cost[i][j] = max(0, (i - 125.0) / 400.0 * i ** (0.5 + diff / 50.0))

        # constraint penalty
        if i > MAX_OCCUPANCY:
            penalty[i] = 60 * (i - MAX_OCCUPANCY + 1) ** 1.2
        elif i < MIN_OCCUPANCY:
            penalty[i] = 60 * (MIN_OCCUPANCY - i + 1) ** 1.2
        else:
            penalty[i] = 0

    return pref_cost, acc1_cost, acc_cost, penalty

def build_global_data(data_dir):
    # family data
    family_choice = np.empty((N_FAMILES, N_CHOICES), dtype=np.int32)
    family_size = np.empty((N_FAMILES,), dtype=np.int32)

    with open(path.join(data_dir, "family_data.csv"), "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            i = int(row[0])
            choices = [int(c) - 1 for c in row[1:N_CHOICES+1]]
            members = int(row[N_CHOICES+1])
            family_size[i] = members
            family_choice[i] = choices
    # cost lut
    pref_cost, acc1_cost, acc_cost, penalty = build_cost_lut(family_size, family_choice)
    return pref_cost, acc1_cost, acc_cost, penalty, family_choice, family_size

PREF_COST, ACC1_COST, ACC_COST, PENALTY, FAMILY_CHOICES, FAMILY_SIZE = build_global_data(DATA_DIR)


@njit
def insertion_cost(f_days, n_days, family_id, d_to, constraint_weight=1):
    """ insertion cost for greedy method
    """
    # preference cost
    pref_cost = PREF_COST[family_id][d_to]

    # accounting cost
    n = n_days[d_to]
    n_new = n + FAMILY_SIZE[family_id]
    if d_to == N_DAYS - 1:
        old_acc_cost = ACC1_COST[n]
        new_acc_cost = ACC1_COST[n_new]
    else:
        old_diff = n - n_days[d_to + 1]
        new_diff = n_new - n_days[d_to + 1]
        old_acc_cost = ACC_COST[n][old_diff + MAX_DIFF]
        new_acc_cost = ACC_COST[n_new][new_diff + MAX_DIFF]

    # constraints penalty
    old_penalty = PENALTY[n]
    new_penalty = PENALTY[n_new]

    return pref_cost + (new_acc_cost - old_acc_cost) + (new_penalty - old_penalty) * constraint_weight


@njit
def total_cost(f_days, c_days, n_days, constraint_weight=1):
    """ cost function for santa-2019
    """
    
    # preference cost
    pref_cost = 0
    for i in range(N_DAYS):
        cost = 0
        for j in range(c_days[i]):
            cost += PREF_COST[f_days[i][j]][i]
        pref_cost += cost

    # accounting cost and constraints penalty
    acc_cost = ACC1_COST[n_days[N_DAYS - 1]]
    penalty = PENALTY[n_days[N_DAYS - 1]]
    for i in range(N_DAYS - 1):
        n = n_days[i]
        diff = n - n_days[i + 1]
        acc_cost += ACC_COST[n][diff + MAX_DIFF]
        penalty += PENALTY[n]
    return pref_cost + acc_cost + penalty * constraint_weight


# utility function for day structure
#
# f_days: family id list for each day
# c_days: size of family id list for each day
# n_days: member size for each day


@njit
def day_insert(f_days, c_days, n_days, day, family_id):
    """ insert family_id to day
    """
    f_days[day][c_days[day]] = family_id
    c_days[day] += 1
    n_days[day] += FAMILY_SIZE[family_id]


@njit
def day_remove(f_days, c_days, n_days, day, pos):
    """ remove family_id from day
    """
    family_id = f_days[day][pos]
    f_days[day][pos] = f_days[day][c_days[day]-1]
    c_days[day] -= 1
    n_days[day] -= FAMILY_SIZE[family_id]
    return family_id


# load/save


def save(filename, f_days, c_days):
    assigned_day = [-1] * N_FAMILES
    for day in range(N_DAYS):
        for family_id in f_days[day][0:c_days[day]]:
            assigned_day[family_id] = day + 1

    with open(filename, "w") as f:
        f.write("family_id,assigned_day\n")
        for family_id in range(N_FAMILES):
            f.write(f"{family_id},{assigned_day[family_id]}\n")


def load(filename):
    f_days = np.zeros((N_FAMILES, MAX_FAMILY_PER_DAY), dtype=np.int32)
    c_days = np.zeros((N_FAMILES,), dtype=np.int32) 
    n_days = np.zeros((N_FAMILES,), dtype=np.int32)
    
    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            family_id, day = int(row[0]), int(row[1]) - 1
            day_insert(f_days, c_days, n_days, day, family_id)
    return f_days, c_days, n_days


# local search


@njit
def generate_initial_solution():
    """ generate the initial solution with greedy method
    """
    f_days = np.zeros((N_FAMILES, MAX_FAMILY_PER_DAY), dtype=np.int32)
    c_days = np.zeros((N_FAMILES,), dtype=np.int32) 
    n_days = np.zeros((N_FAMILES,), dtype=np.int32)

    # sort by family member size
    family_size = np.empty((N_FAMILES,), dtype=np.int32)
    order = np.empty((N_FAMILES,), dtype=np.int32)
    for family_id in range(N_FAMILES):
        family_size[family_id] = -FAMILY_SIZE[family_id]
    order = np.argsort(family_size)

    # insert with greedy method
    for i in range(N_FAMILES):
        family_id = order[i]
        min_cost = DBL_MAX
        best_day = -1
        for day in range(N_DAYS):
            cost = insertion_cost(f_days, n_days, family_id, day)
            if cost < min_cost:
                min_cost = cost
                best_day = day
        day_insert(f_days, c_days, n_days, best_day, family_id)
    return f_days, c_days, n_days


@njit
def generate_neighbor_solution(f_days, c_days, n_days, step):
    """ generate a neighbor solution with destroy and repair method
    """

    # randomly destroy day/family
    day_changed = Dict.empty(numba.int32, numba.int8)
    if np.random.uniform(0, 1) > 0.5:
        method = 0
        if step % 2 == 0:
            day_samples = 1 + np.random.randint(0, 16)
        else:
            day_samples = 2 + np.random.randint(0, 4)
        if step % 3 == 0:
            family_sample_max = 1 + np.random.randint(0, 16)
        else:
            family_sample_max = 2 + np.random.randint(0, 4)
    else:
        method = 1
        if step % 2 == 0:
            day_samples = 1 + np.random.randint(0, 4)
        else:
            day_samples = 2 + np.random.randint(0, 16)
        family_sample_max = 1
    removed_families = np.empty((day_samples * family_sample_max,), dtype=np.int32)
    removed_count = 0

    if method == 0:
        for i in range(day_samples):
            day = np.random.randint(0, N_DAYS)
            family_sample = 1 + np.random.randint(0, family_sample_max)
            for j in range(family_sample):
                family_count = c_days[day]
                if family_count > 0:
                    pos = np.random.randint(0, family_count)
                    family_id = day_remove(f_days, c_days, n_days, day, pos)
                    removed_families[removed_count] = family_id
                    removed_count += 1
                    day_changed[day] = 1
    else:
        for i in range(day_samples):
            day = np.random.randint(0, N_DAYS)
            family_count = c_days[day]
            if family_count > 0:
                pos = np.random.randint(0, family_count)
                family_id = day_remove(f_days, c_days, n_days, day, pos)
                removed_families[removed_count] = family_id
                removed_count += 1
                day_changed[day] = 1

    # repair with greedy method
    
    removed_families = removed_families[:removed_count]
    if np.random.uniform(0, 1) > 0.5:
        np.random.shuffle(removed_families)
    else:
        family_size = -FAMILY_SIZE[removed_families]
        removed_families = removed_families[family_size.argsort()]

    for family_id in removed_families:
        min_cost = DBL_MAX
        best_day = -1
        for i in range(N_CHOICES):
            day = FAMILY_CHOICES[family_id][i]
            cost = insertion_cost(f_days, n_days, family_id, day)
            if cost < min_cost:
                min_cost = cost
                best_day = day
        day_insert(f_days, c_days, n_days, best_day, family_id)
        day_changed[best_day] = 1

    return day_changed


# hill climbing

SAVE_INTERVAL = 5

def hill_climbing(f_days, c_days, n_days, lahc_size=200, stuck_time=120, limit_time=None):
    start_at = last_updated = last_saved = time()
    prev_cost = best_cost = total_cost(f_days, c_days, n_days)
    step = 0
    hist = np.empty((lahc_size,), dtype=np.float32) # history list for late acceptance hill climbing
    hist[:] = prev_cost
    t_f_days, t_c_days, t_n_days = f_days.copy(), c_days.copy(), n_days.copy()

    while True: # loooooooooop
        day_changed = generate_neighbor_solution(t_f_days, t_c_days, t_n_days, step)
        cost = total_cost(t_f_days, t_c_days, t_n_days)
        step += 1
        hist_index = step % lahc_size
        if cost != prev_cost and (cost < prev_cost or cost < hist[hist_index]):
            # update
            for day in day_changed.keys():
                c = max(c_days[day], t_c_days[day])
                f_days[day, :c] = t_f_days[day, :c]
                c_days[day], n_days[day] = t_c_days[day], t_n_days[day]
            prev_cost = cost
            hist[hist_index] = cost
            
            now = time()
            if cost < best_cost and now - last_saved > SAVE_INTERVAL:
                best_cost = cost
                print(f"{step}: {cost}, {int(step / (now - start_at))} step/s, {int((now - start_at) / 60)} min")
                # save submission file
                save("best.csv", f_days, c_days)
                last_saved = now
            last_updated = now
        else:
            # revert
            for day in day_changed.keys():
                c = max(c_days[day], t_c_days[day])
                t_f_days[day, :c] = f_days[day, :c]
                t_c_days[day], t_n_days[day] = c_days[day], n_days[day]

        if step % 10000 == 0:
            now = time()
            if now - last_updated > stuck_time:
                print("!! I AM STUCK !!")
                break
            if limit_time is not None and now - start_at > limit_time:
                print("!! TIME LIMIT !!")
                break

if __name__ == "__main__":
    np.random.seed(71)
    f_days, c_days, n_days = generate_initial_solution()
    # large lahc_size gives a good score but takes a lot of time
    # limit_time=3600*6 is for kaggle kernel timeout
    hill_climbing(f_days, c_days, n_days, lahc_size=100, stuck_time=120, limit_time=3600*6)
    f_days, c_days, n_days = load("best.csv")
    print("Done: ", total_cost(f_days, c_days, n_days))
    save("submission.csv", f_days, c_days)
