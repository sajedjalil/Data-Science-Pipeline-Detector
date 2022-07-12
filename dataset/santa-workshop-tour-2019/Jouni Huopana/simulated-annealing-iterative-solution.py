# Simulated annealing iteration
# Partial data is optimised in a loop until solution is found

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numba import njit
from scipy import optimize

data = pd.read_csv("/kaggle/input/santa-workshop-tour-2019/family_data.csv", index_col='family_id')
submission = pd.read_csv("/kaggle/input/santa-workshop-tour-2019/sample_submission.csv", index_col='family_id')

print(data.head())

N_DAYS = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125
family_size = data.n_people.values
days_array = np.arange(N_DAYS, 0, -1)
choice_dict = data.loc[:, 'choice_0': 'choice_9'].T.to_dict()
choice_array_num = np.full((data.shape[0], N_DAYS + 1), -1)

for i, choice in enumerate(data.loc[:, 'choice_0': 'choice_9'].values):
    for d, day in enumerate(choice):
        choice_array_num[i, day] = d

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


#Collected from fast solution notebook, modified cost per occupancy error
@njit
def cost_function(prediction, penalties_array, family_size, days):
    penalty = 0
    #
    # We'll use this to count the number of people scheduled each day
    daily_occupancy = np.zeros((len(days)+1))
    N = family_size.shape[0]
    #
    # Looping over each family; d is the day, n is size of that family, 
    # and choice is their top choices
    for i in range(N):
        # add the family member count to the daily occupancy
        n = family_size[i]
        d = prediction[i]
        choice = choice_array_num[i]
        #
        daily_occupancy[d] += n
        #
        # Calculate the penalty for not getting top preference
        penalty += penalties_array[n, choice[d]]
    #
    # for each date, check total occupancy
    #  (using soft constraints instead of hard constraints)
    relevant_occupancy = daily_occupancy[1:]
    #Penalty increase for every occupancy error
    penalty += np.sum(relevant_occupancy[relevant_occupancy>MAX_OCCUPANCY])*2000
    penalty += np.sum(125-relevant_occupancy[relevant_occupancy<MIN_OCCUPANCY])*2000
    # Calculate the accounting cost
    # The first day (day 100) is treated special
    init_occupancy = daily_occupancy[days[0]]
    accounting_cost = (init_occupancy - 125.0) / 400.0 * init_occupancy**(0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)
    #
    # Loop over the rest of the days, keeping track of previous count
    yesterday_count = init_occupancy
    for day in days[1:]:
        today_count = daily_occupancy[day]
        diff = np.abs(today_count - yesterday_count)
        accounting_cost += max(0, (today_count - 125.0) / 400.0 * today_count**(0.5 + diff / 50.0))
        yesterday_count = today_count
    #
    penalty += accounting_cost
    #
    return penalty


lookup = data[["choice_0","choice_1","choice_2","choice_3","choice_4","choice_5","choice_6","choice_7","choice_8","choice_9"]].values

x0 = lookup[np.arange(5000),np.random.choice(10, 5000)]
x0_t = np.copy(x0)

def someCost(x):
    x0[fam]=lookup[fam,x.astype(int)]
    pcost = cost_function(x0, penalties_array, family_size, days_array)
    return pcost


def occupants(prediction, penalties_array, family_size, days):
    # We'll use this to count the number of people scheduled each day
    daily_occupancy = np.zeros((len(days)+1))
    N = family_size.shape[0]
    for i in range(N):
        # add the family member count to the daily occupancy
        n = family_size[i]
        d = prediction[i]
        choice = choice_array_num[i]
        #
        daily_occupancy[d] += n
    return daily_occupancy

# Select number of families to optimize with simulated annealing
# Increase this for faster convergance but for slower solution
selec = 21

# Choice index
bounds=[(0, 9)]*selec

i=0
best = 10.0e12
while True:
    #Select random families
    fam = np.random.choice(5000, selec, replace=False)
    #Optimise those
    res = optimize.dual_annealing(someCost, bounds, maxiter=180)
    x0_t[fam] = lookup[fam,res.x.astype(int)]
    new_c = cost_function(x0_t, penalties_array, family_size, days_array)
    #Compare old solution to the new and save if better
    if new_c<best:
        x0=np.copy(x0_t)
        best = new_c
    del res
    test=occupants(x0, penalties_array, family_size, days_array)
    test=test[1:]
    test = sum((test<125)+(test>300))
    if i%10==0:
        print('Iter:',i,'Occupancy error:',test,'Current cost:',round(new_c), 'Best:',round(best))
    i=i+1
    #Write result if occupancy ok
    if test==0:
        submission['assigned_day'] = x0
        submission.to_csv(f'submission'+str(round(new_c))+'.csv')
        print("Saved.")
        break
