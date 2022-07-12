import numpy as np
import pandas as pd
import random
from scipy.optimize import linear_sum_assignment

data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col='family_id')
submission = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/sample_submission.csv', index_col='family_id')

family_size_dict = data[['n_people']].to_dict()['n_people']
# with open('n.txt','w') as f:
#     for i in range(5000):
#         f.write(str(family_size_dict[i])+' ')

cols = [f'choice_{i}' for i in range(10)]
choice = np.array(data[cols])

def cal_cost(n):
    arr = np.zeros((11,))
    arr[0] = 0
    arr[1] = 50
    arr[2] = 50 + 9 * n
    arr[3] = 100 + 9 * n
    arr[4] = 200 + 9 * n
    arr[5] = 200 + 18 * n
    arr[6] = 300 + 18 * n
    arr[7] = 300 + 36 * n
    arr[8] = 400 + 36 * n
    arr[9] = 500 + 235 * n
    arr[10] = 500 + 434 * n
    return arr

cost = np.zeros((5000,100),dtype='int32')
for i in range(5000):
    for j in range(100):
        cost[i,j]=cal_cost(family_size_dict[i])[10]
for i in range(5000):
    for j in range(10):#第几个计划
        c = choice[i,j]-1#该计划在哪天
        n = family_size_dict[i]#人数
        cost[i,c] = cal_cost(n)[j]

print(cost.shape)

res = np.zeros((5000,),dtype='int32')
day_occupy = np.zeros((100,))

cost = np.transpose(cost)
# cost = np.repeat(cost,50,axis=1)
print(cost.shape)

while 0 in res:
    row_ind,col_ind=linear_sum_assignment(cost)
    # print(row_ind)#开销矩阵对应的行索引（第几天
    # print(col_ind)#对应行索引的最优指派的列索引（第几个家庭
    # print(day_occupy)
    for i in range(len(col_ind)):
        if day_occupy[i] <= 300:
            res[col_ind[i]] = int(i+1)
            cost[:,col_ind[i]] += 100000000
            day_occupy[i] += family_size_dict[col_ind[i]]

print(res)

submission['assigned_day'] = res

choice_dict = data[cols].to_dict()
# print(choice_dict)

N_DAYS = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125

# from 100 to 1
days = list(range(N_DAYS,0,-1))

family_size_ls = list(family_size_dict.values())
choice_dict_num = [{vv:i for i, vv in enumerate(di.values())} for di in choice_dict.values()]

def cost_function(prediction):
    penalty = 0

    # We'll use this to count the number of people scheduled each day
    daily_occupancy = {k: 0 for k in days}
    for f, d in enumerate(prediction):
        n = family_size_dict[f]
        choice_0 = choice_dict['choice_0'][f]
        choice_1 = choice_dict['choice_1'][f]
        choice_2 = choice_dict['choice_2'][f]
        choice_3 = choice_dict['choice_3'][f]
        choice_4 = choice_dict['choice_4'][f]
        choice_5 = choice_dict['choice_5'][f]
        choice_6 = choice_dict['choice_6'][f]
        choice_7 = choice_dict['choice_7'][f]
        choice_8 = choice_dict['choice_8'][f]
        choice_9 = choice_dict['choice_9'][f]

        # add the family member count to the daily occupancy
        daily_occupancy[d] += n

        # Calculate the penalty for not getting top preference
        if d == choice_0:
            penalty += 0
        elif d == choice_1:
            penalty += 50
        elif d == choice_2:
            penalty += 50 + 9 * n
        elif d == choice_3:
            penalty += 100 + 9 * n
        elif d == choice_4:
            penalty += 200 + 9 * n
        elif d == choice_5:
            penalty += 200 + 18 * n
        elif d == choice_6:
            penalty += 300 + 18 * n
        elif d == choice_7:
            penalty += 300 + 36 * n
        elif d == choice_8:
            penalty += 400 + 36 * n
        elif d == choice_9:
            penalty += 500 + 36 * n + 199 * n
        else:
            penalty += 500 + 36 * n + 398 * n

    # for each date, check total occupancy
    #  (using soft constraints instead of hard constraints)
    for v in daily_occupancy.values():
        if (v > MAX_OCCUPANCY) or (v < MIN_OCCUPANCY):
            penalty += 100000000
            return penalty
    # Calculate the accounting cost
    # The first day (day 100) is treated special
    accounting_cost = (daily_occupancy[days[0]] - 125.0) / 400.0 * daily_occupancy[days[0]] ** (0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)

    # Loop over the rest of the days, keeping track of previous count
    yesterday_count = daily_occupancy[days[0]]
    for day in days[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += max(0, (daily_occupancy[day] - 125.0) / 400.0 * daily_occupancy[day] ** (0.5 + diff / 50.0))
        yesterday_count = today_count

    penalty += accounting_cost

    return penalty

best = submission['assigned_day'].tolist()

start_score = cost_function(best)
print(start_score)

new = best.copy()

for i in range(10):
    fam = list(range(5000))
    random.shuffle(fam)
    # loop over each family
    for fam_id in fam:
        # loop over each family choice
        for pick in range(7):
            day = choice_dict[f'choice_{pick}'][fam_id]
            temp = new.copy()
            temp[fam_id] = day # add in the new pick
            temp_cost = cost_function(temp)
            if temp_cost < start_score:
                new = temp.copy()
                start_score = temp_cost
                print('Score: '+str(start_score), end='\r', flush=True)

    submission['assigned_day'] = new
    score = start_score
    submission.to_csv(f'submission_{int(score)}.csv')
    print(f'Score: {score}')