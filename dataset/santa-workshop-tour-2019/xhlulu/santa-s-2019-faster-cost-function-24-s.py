"""
# About this kernel

The `cost_function` in this kernel is roughly 600x faster compared to the original kernel. 
Each function call takes roughly 24 µs.

## Quick Start

1. Import this utility file: File > Add utility script > Search Notebooks > *Type this notebook name*

2. Copy the code below to get started:
```
# Imports
import pandas as pd
import numpy as np

# The name of the kernel might change, so update this if needed
from santa_s_2019_faster_cost_function_24_s import build_cost_function

# Load Data
base_path = '/kaggle/input/santa-workshop-tour-2019/'
data = pd.read_csv(base_path + 'family_data.csv', index_col='family_id')
submission = pd.read_csv(base_path + 'sample_submission.csv', index_col='family_id')

# Build your "cost_function"
cost_function = build_cost_function(data)

# Run it on default submission file
best = submission['assigned_day'].values
start_score = cost_function(best)
```

A longer example is provided at the end.


## Note

Starting in V12, I decided to make this an utility script instead of a regular notebook.
I think this is a better use of this kernel, since you can now directly import this into
your project and use it just like an API, instead of copy-pasting the lengthy code.

I think that make this into a script forces me to keep the code cleaner.

## Reference

* (Excellent) Original Kernel: https://www.kaggle.com/inversion/santa-s-2019-starter-notebook
* First kernel that had the idea to use Numba: https://www.kaggle.com/nickel/250x-faster-cost-function-with-numba-jit
* Another great cost function optimization: https://www.kaggle.com/sekrier/fast-scoring-using-c-52-usec
* More modular output for intermediate function: https://www.kaggle.com/nickel/santa-s-2019-fast-pythonic-cost-23-s
"""

import os
from functools import partial

from numba import njit
import numpy as np
import pandas as pd
from tqdm import tqdm


## Intermediate Helper Functions
def _build_choice_array(data, n_days):
    choice_matrix = data.loc[:, 'choice_0': 'choice_9'].values
    choice_array_num = np.full((data.shape[0], n_days + 1), -1)

    for i, choice in enumerate(choice_matrix):
        for d, day in enumerate(choice):
            choice_array_num[i, day] = d
    
    return choice_array_num


def _precompute_accounting(max_day_count, max_diff):
    accounting_matrix = np.zeros((max_day_count+1, max_diff+1))
    # Start day count at 1 in order to avoid division by 0
    for today_count in range(1, max_day_count+1):
        for diff in range(max_diff+1):
            accounting_cost = (today_count - 125.0) / 400.0 * today_count**(0.5 + diff / 50.0)
            accounting_matrix[today_count, diff] = max(0, accounting_cost)
    
    return accounting_matrix


def _precompute_penalties(choice_array_num, family_size):
    penalties_array = np.array([
        [
            0,
            50,
            50 + 9 * n,
            100 + 9 * n,
            200 + 9 * n,
            200 + 18 * n,
            300 + 18 * n,
            300 + 36 * n,
            400 + 36 * n,
            500 + 36 * n + 199 * n,
            500 + 36 * n + 398 * n
        ]
        for n in range(family_size.max() + 1)
    ])
    
    penalty_matrix = np.zeros(choice_array_num.shape)
    N = family_size.shape[0]
    for i in range(N):
        choice = choice_array_num[i]
        n = family_size[i]
        
        for j in range(penalty_matrix.shape[1]):
            penalty_matrix[i, j] = penalties_array[n, choice[j]]
    
    return penalty_matrix


@njit
def _compute_cost_fast(prediction, family_size, days_array, 
                       penalty_matrix, accounting_matrix, 
                       MAX_OCCUPANCY, MIN_OCCUPANCY, N_DAYS):
    """
    Do not use this function. Please use `build_cost_function` instead to 
    build your own "cost_function".
    """
    N = family_size.shape[0]
    # We'll use this to count the number of people scheduled each day
    daily_occupancy = np.zeros(len(days_array)+1, dtype=np.int64)
    penalty = 0
    
    # Looping over each family; d is the day, n is size of that family
    for i in range(N):
        n = family_size[i]
        d = prediction[i]
        
        daily_occupancy[d] += n
        penalty += penalty_matrix[i, d]

    # for each date, check total occupancy 
    # (using soft constraints instead of hard constraints)
    # Day 0 does not exist, so we do not count it
    relevant_occupancy = daily_occupancy[1:]
    incorrect_occupancy = np.any(
        (relevant_occupancy > MAX_OCCUPANCY) | 
        (relevant_occupancy < MIN_OCCUPANCY)
    )
    
    penalty = 100000000

    # Calculate the accounting cost
    # The first day (day 100) is treated special
    init_occupancy = daily_occupancy[days_array[0]]
    accounting_cost = (init_occupancy - 125.0) / 400.0 * init_occupancy**(0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)
    
    # Loop over the rest of the days_array, keeping track of previous count
    yesterday_count = init_occupancy
    for day in days_array[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += accounting_matrix[today_count, diff]
        yesterday_count = today_count

    return penalty, accounting_cost, daily_occupancy


def build_cost_function(data, N_DAYS=100, MAX_OCCUPANCY=300, MIN_OCCUPANCY=125):
    """
    data (pd.DataFrame): 
        should be the df that contains family information. Preferably load it from "family_data.csv".
    """
    family_size = data.n_people.values
    days_array = np.arange(N_DAYS, 0, -1)

    # Precompute matrices needed for our cost function
    choice_array_num = _build_choice_array(data, N_DAYS)
    penalty_matrix = _precompute_penalties(choice_array_num, family_size)
    accounting_matrix = _precompute_accounting(max_day_count=MAX_OCCUPANCY, max_diff=MAX_OCCUPANCY)
    
    # Partially apply `_compute_cost_fast` so that the resulting partially applied
    # function only requires prediction as input. E.g.
    # Non partial applied: score = _compute_cost_fast(prediction, family_size, days_array, ...)
    # Partially applied: score = cost_function(prediction)
    def cost_function(prediction):
        penalty, accounting_cost, daily_occupancy = _compute_cost_fast(
            prediction=prediction,
            family_size=family_size, 
            days_array=days_array, 
            penalty_matrix=penalty_matrix, 
            accounting_matrix=accounting_matrix,
            MAX_OCCUPANCY=MAX_OCCUPANCY,
            MIN_OCCUPANCY=MIN_OCCUPANCY,
            N_DAYS=N_DAYS
        )
        
        return penalty + accounting_cost
    
    return cost_function
    

# LONGER EXAMPLE STARTS HERE
if __name__ == '__main__':
    import timeit
    
    # Load Data
    base_path = '/kaggle/input/santa-workshop-tour-2019/'
    data = pd.read_csv(base_path + 'family_data.csv', index_col='family_id')
    submission = pd.read_csv(base_path + 'sample_submission.csv', index_col='family_id')

    # Build our cost_function
    cost_function = build_cost_function(data)

    # Start with the sample submission values
    best = submission['assigned_day'].values


    # Let's see how fast it is:
    function_call_times = timeit.repeat(lambda: cost_function(best), repeat=10, number=20000)
    mean_call_time = np.array(function_call_times[3:]).mean() / 20000
    print(f"cost_function takes {mean_call_time:.3e} seconds to run")

    # We can now proceed with the optimization
    choice_matrix = data.loc[:, 'choice_0': 'choice_9'].values
    start_score = cost_function(best)
    new = best.copy()
    # loop over each family
    for fam_id in tqdm(range(len(best))):
        # loop over each family choice
        for pick in range(10):
            day = choice_matrix[fam_id, pick]
            temp = new.copy()
            temp[fam_id] = day # add in the new pick
            score = cost_function(temp)
            if score < start_score:
                new = temp.copy()
                start_score = score

    score = cost_function(new)
    print(f'Score: {score}')
    submission['assigned_day'] = new
    submission.to_csv(f'submission_{score}.csv')
                      