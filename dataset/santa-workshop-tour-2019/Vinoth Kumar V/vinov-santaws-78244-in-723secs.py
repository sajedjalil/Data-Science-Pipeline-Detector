# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

data = pd.read_csv('../input/santa-workshop-tour-2019/family_data.csv')

# Parameters to compute preference cost
L1 = [0,50,50,100,200,200,300,300,400,500,500]
L2 = [0,0,9,9,9,18,18,36,36,235,434]


days = range(1,101)
daysPpl = {k:0 for k in days}
m = 5000  # len(data)

# -------------------------------------------------------
# Returns a dictionary  {day -> no of ppl scheduled for that day}
# gives an ideal distribution of ppl across days 1 to 100
# that gives somewhat optimal value for adminCost
# Note: scheduling 210 ppl on each day would be least admincost
#       but that might have adverse effect on preference cost
# -------------------------------------------------------
def headCountDensityFunc():
    f1 = {1:286,2:285,3:280,4:265,5:265,6:245,7:235,8:215,9:205,10:205,11:200,12:200,13:200,14:205}
    f2 = {1:270,2:260,3:250,4:225,5:225,6:185,7:170,8:150,9:135,10:135,11:130,12:130,13:130,14:135}
    dp = {k:0 for k in days}
    a1 = 295
    b1 = 250
    c1 = 0
    for i in range(3,95,7):
        c1 += 1
        a1 = f1[c1]
        b1 = f2[c1]
        dp[i] = a1 - 10
        dp[i+1] = a1 
        dp[i+2] = a1 - 10
        dp[i+3] = b1 + 10
        dp[i+4] = b1 - 5
        dp[i+5] = b1 - 5
        dp[i+6] = b1 + 8
        
    dp[1] = 298
    dp[2] = 280 
    dp[100] = 125
    s1 = sum(dp.values())   
    diff = 21003 - s1
    c1 = 3
    while diff > 0:
        if dp[c1] < 298:
            dp[c1] += 1
            diff -= 1
        c1 +=1
        if c1 > 54:
            c1 = 3

    return dp

# -------------------------------------------------------
# Get Choice Number (0 thru 10 - 10 = none of the choice 
# of family i (assumed sorted by family_id), choice day
# -------------------------------------------------------
def getChoiceNumber(i,cday):
    chNum = 10
    for j in range(1,11):
        if data.iloc[i,j]==cday:
            chNum = j - 1
            break
     
    return chNum

# -------------------------------------------------------
# Calculates Admin Cost for a given distribution of ppl across days
# -------------------------------------------------------
def adminCost(nd):
    #day 100 is first entry
    ni = nd[100]
    acost = ((ni - 125)/400) * (ni ** 0.5)
    prev = ni
    for i in range(99,0,-1):
        ni = nd[i]
        diff = abs(ni - prev)
        s1 = ((ni - 125)/400) * (ni ** (0.5 + diff/50))
        acost += s1
        prev = ni
    #end-for
    return acost

# -------------------------------------------------------
#  Calculates cost function for given assigments
#  x = list of assigned days - implicily ordered by family id
# -------------------------------------------------------
def costFunction(x):
    daysPpl = dict.fromkeys(days,0)
    pcost = 0
    for i in range(0,m):
        cday = x[i]
        #famId = i
        n = data.iloc[i,11]
        #get choice number (0-9) default = 10
        ch = getChoiceNumber(i,cday)
        #caculate preference cost
        pcost += L1[ch] + L2[ch]*n
        #add n to daysPpl dict for that day
        daysPpl[cday] += n
    #end-for
    acost = adminCost(daysPpl)
    totalCost = acost + pcost
    #print('acost='+str(acost)+',pcost='+str(pcost)+',total='+str(totalCost))
    return totalCost

# -------------------------------------------------------
# Calculates Preference Cost Delta of
# of moving a family of size n, from day1 to day2
# -------------------------------------------------------
def calculatePreferenceCostDelta(i,n,day1, day2):
    ch1 = getChoiceNumber(i,day1)
    pcostOrig = L1[ch1] + L2[ch1]*n
    ch2 = getChoiceNumber(i,day2)
    pcostNew = L1[ch2] + L2[ch2]*n
    diff = pcostNew - pcostOrig
    return diff
# -------------------------------------------------------
# Calculates Admin Cost Delta of
# of moving a family of size n, from day1 to day2
# -------------------------------------------------------
def calculateAdminCostDelta(n,day1,day2):
    #moving a family of size n from day1 to day2
    #calc terms for day1-1 and day1
    #Calcate original 4 terms for day1, day-1, day2, day2-1
    if (day1 == 1):
        n0 = 125
    else:
        n0 = daysPpl[day1-1]
    n1 = daysPpl[day1]
    if day1 == 100:
        n2 = n1
    else:
        n2 = daysPpl[day1+1]
    
    diff = abs(n0-n1)/50
    t1 = ((n0 - 125)/400) * (n0 ** (0.5 + diff))
    diff = abs(n1-n2)/50
    t1 += ((n1 - 125)/400) * (n1 ** (0.5 + diff))

    if (day2 == 1):
        n0 = 125
    else:
        n0 = daysPpl[day2-1]
    n1 = daysPpl[day2]
    if day2 == 100:
        n2 = n1
    else:
        n2 = daysPpl[day2+1]

    
    diff = abs(n0-n1)/50
    t1 += ((n0 - 125)/400) * (n0 ** (0.5 + diff))
    diff = abs(n1-n2)/50
    t1 += ((n1 - 125)/400) * (n1 ** (0.5 + diff))

    
    #Calcate New 4 terms for day1, day-1, day2, day2-1
    if (day1 == 1):
        n0 = 125
    else:
        n0 = daysPpl[day1-1] 
    n1 = daysPpl[day1] - n
    if day1 == 100:
        n2 = n1
    else:
        n2 = daysPpl[day1+1]
    
    diff = abs(n0-n1)/50
    t2 = ((n0 - 125)/400) * (n0 ** (0.5 + diff))
    diff = abs(n1-n2)/50
    t2 += ((n1 - 125)/400) * (n1 ** (0.5 + diff))

    if (day2 == 1):
        n0 = 125
    else:
        n0 = daysPpl[day2-1]
    n1 = daysPpl[day2] + n
    if day2 == 100:
        n2 = n1
    else:
        n2 = daysPpl[day2+1]

    
    diff = abs(n0-n1)/50
    t2 += ((n0 - 125)/400) * (n0 ** (0.5 + diff))
    diff = abs(n1-n2)/50
    t2 += ((n1 - 125)/400) * (n1 ** (0.5 + diff))
    
    diff = t2 - t1
    return diff

def calculateCostDelta(i,n,day1, day2):
    adiff = calculateAdminCostDelta(n,day1,day2) 
    pdiff = calculatePreferenceCostDelta(i,n,day1,day2)
    return adiff + pdiff
    

def analyzeSol(x):
    c1 = range(0,11)
    y1=pd.DataFrame(c1,columns=['choiceNum'])
    y1['famCount']=0
    y1['pplCount']=0
    m = len(x)
    for i in range(0,m):
        cday = x[i]
        chNum = getChoiceNumber(i,cday)
        nPpl = data.iloc[i,11]
        y1.iloc[chNum,1] = y1.iloc[chNum,1] + 1
        y1.iloc[chNum,2] = y1.iloc[chNum,2] + nPpl
    #end-for
    print("---- Solution assignment summary ---")
    print(y1.head(20))
    print("-------------------")
    return y1

timeStart = time.time()
# Target distribution of ppl across 100 days
ndist = headCountDensityFunc()

# Add additional fields to data for ease of computation
data['assigned_day']=0   #data.iloc[i,12]
data['assigned_choice']=10  #data.iloc[i,13]


data = data.sort_values('n_people',ascending=True)

# Find low assignment probablity days
for j in range(1,3):
    for i in range(0,m):
        if data.iloc[i,13] < 10:
            continue
        choice_day = data.iloc[i,j]
        n = data.iloc[i,11]
        currPpl = daysPpl[choice_day]
        newPpl = currPpl + n
        if newPpl > 220:
            continue
        
        data.iloc[i,12] = choice_day
        data.iloc[i,13] = j - 1
        daysPpl[choice_day] = newPpl

lowDays = []
dp2 = sorted(daysPpl, key=daysPpl.get, reverse=False)
for i in dp2:
    if daysPpl[i] <= 130:
        lowDays.append(i)     

# Reset all assignments
data['assigned_day']=0   #data.iloc[i,12]
data['assigned_choice']=10  #data.iloc[i,13]
daysPpl = {k:0 for k in days}

# Iteration #1 - Fill only low probablity days (upto 125) 

data = data.sort_values('n_people',ascending=False)
max_family_count = 4 #per iteration
max_choice = 5
for k in range(0,20):
    asgn_count = 0
    for d in lowDays:
        asgn_family_count = 0
        if daysPpl[d] >= 125:
            continue
        for j in range(1,max_choice+1):
            for i in range(0,m):
                if data.iloc[i,13] < 10:
                    #some day is already assigned for this family
                    continue
                
                choice_day = data.iloc[i,j]
                if not choice_day == d:
                    continue
                
                n = data.iloc[i,11] # no of ppl in the family
                curr_ppl = daysPpl[choice_day]
                if curr_ppl >= 125:
                    break
                
                #assign this day to the family 'i'
                data.iloc[i,12] = choice_day
                data.iloc[i,13] = j - 1
                daysPpl[choice_day] += n
                asgn_family_count += 1
                asgn_count += 1
                if asgn_family_count >= max_family_count:
                    break
            #end-for i
            if daysPpl[choice_day] >= 125:
                break
            if asgn_family_count >= max_family_count:
                break
            
        #end-for j
    #end-for d
    if asgn_count == 0:
        break

print('End of iteration 1')        

# Iteration #2 - Fill all days up to max defined by ndist
data = data.sort_values('n_people',ascending=True)

max_choice = 4
delta = 0
loc_max = 12
for k in range(1,60):
    asgn_count = 0
    loc_max += 6
    for j in range(1,max_choice+1):
        for i in range(0,m):
            if data.iloc[i,13] < 10:
                #already assigned
                continue
            choice_day = data.iloc[i,j]
            n = data.iloc[i,11] # no of ppl in the family
            max_ppl = min(loc_max,ndist[choice_day])
            new_ppl = daysPpl[choice_day] + n
            if new_ppl <= max_ppl:
                data.iloc[i,12] = choice_day
                data.iloc[i,13] = j - 1
                daysPpl[choice_day] = new_ppl
                asgn_count += 1
         #end-for i
            
    #end-for j
    if asgn_count == 0:
        break
    
    
print('End of iteration 2')        
print(daysPpl)



# final iteration - just assign closest choice
max_choice = 10       
delta = 20
for i in range(0,m):
    if data.iloc[i,13] < 10:
        continue
    n = data.iloc[i,11]
    for j in range(1,max_choice+1):
        choice_day = data.iloc[i,j]
        curr_ppl = daysPpl[choice_day]
        new_ppl = curr_ppl + n
        max_ppl = min(ndist[choice_day] + delta,300)
        if new_ppl <= max_ppl:
            data.iloc[i,12] = choice_day
            data.iloc[i,13] = j - 1
            daysPpl[choice_day] += n
            break


print('End of iteration 3')        
print(daysPpl)
timeEnd = time.time()
timeDiff = timeEnd- timeStart
print('Total time taken Part #1 = '+str(timeDiff)+' seconds')

best = data['assigned_day'].values
analyzeSol(best)
cost = costFunction(best)
print('Starting cost = '+str(cost))

file_sub = pd.DataFrame(data[['family_id','assigned_day']])
file_sub['assigned_day'] = best
fname = f'submission_init.csv'
file_sub.to_csv(fname,index=False)


print('Begining optimization process')
print('Starting cost = '+str(cost))

MIN = 125
MAX = 300
best_score = cost
new_score = best_score
new = best.copy()

for k in range(1,30):
    strk = '#### k = '+str(k)
    improve_count = 0
    for i in range(k,m):
        n = data.iloc[i,11]
        stri = strk+', i = '+str(i)
        choice_day = new[i]
        for j in range(1,10):
            new_choice = data.iloc[i,j]
            if choice_day == new_choice:
                continue
            # moving day for this customer from choice_day to new_choice
            x = daysPpl[choice_day] - n
            y = daysPpl[new_choice] + n
            
            if x < MIN or y > MAX:
                continue
            
            cost_diff = calculateCostDelta(i,n,choice_day, new_choice)
            if cost_diff < 0:
                prev_choice = new[i]
                new[i] = new_choice
                new_score = costFunction(new)
                if new_score < best_score:
                    best_score = new_score
                    best = new.copy()
                    daysPpl[choice_day] -= n
                    daysPpl[new_choice] +=n 
                    improve_count += 1
                    print(stri+' - score = '+str(best_score))
                    break
                else:
                    #revert back assignment
                    new[i] = prev_choice
            #end-if
        #end-for j
    #end-for i
    print(' --------- End of iteration ------ '+strk)
    print('Improve count = '+str(improve_count))
    data['assigned_day'] = best
    file_sub['assigned_day'] = best
    fname = f'submission_{best_score}.csv'
    file_sub.to_csv(fname,index=False)
    
    if improve_count == 0 and k > 20:
        break
    
#write to file


file_sub['assigned_day'] = best
fname = f'submission_{best_score}.csv'
file_sub.to_csv(fname,index=False)

timeEnd = time.time()
timeDiff = timeEnd- timeStart
print('Total time taken = '+str(timeDiff)+' seconds')

# %% [code]
