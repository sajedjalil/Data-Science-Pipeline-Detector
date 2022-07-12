# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        pass

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

def mutation(chromo):
    x = np.random.randint(family_no)
    x = 100**x
    upper = chromo//x
    lower = chromo%x
    upper = ((upper//100)*100+np.random.randint(100))
    return upper+lower
   
    
def cost_function(chromo):
    days_visitors = [0 for i in range(101)]
    cost = 0
    for i in range(family_no):
        day = chromo % 100
        expected = False
        for j in range(len(family_chioce[i])):
            if day == family_chioce[i][j]-1:
                cost += cost_matrix[family_member[i]][j]
                expected = True
                break
            if not expected:
                cost += cost_matrix[family_member[i]][-1]
    last_day = days_visitors[1]
    for day in days_visitors[1:]:
        cost += (day-125)/400 * pow(day, 0.5+abs(day-last_day)/50)
    for day in days_visitors:
        if day > 300 or day < 125:
            cost += 100000000000
    return cost

def crossover(chromo1, chromo2):
    x = np.random.randint(family_no)
    x = 100**x
    c11 = chromo1 // x
    c12 = chromo1 % x
    c21 = chromo2 // x
    c22 = chromo2 % x
    return c11*x+c22, c21*x+c12
    
                        
        
#hyper parameters
population_no = 100
offspring_no = 100
epochs      = 100


fpath = '/kaggle/input/santa-workshop-tour-2019/family_data.csv'
data = pd.read_csv(fpath, index_col='family_id')
family_no      = data.shape[0]
family_chioce  = data[["choice_0", "choice_1", "choice_2", "choice_3", "choice_4", "choice_5", "choice_6", "choice_7", "choice_8", "choice_9", ]].to_numpy()
family_member = data["n_people"].to_numpy()

cost_matrix = [
    [0,
     50,
     50 + 9*i,
     100+ 9*i,
     200+ 9*i,
     200+18*i, 
     300+18*i, 
     300+36*i, 
     400+36*i, 
     500+36*i, 
     500+36*i+398*i
    ]
    for i in range(max(family_member)+1)
]

population = []
for i in range(population_no):
    chromo=0
    for j in range(family_no):
        chromo += np.random.randint(100)+1
        chromo *= 100
    chromo =  chromo // 100
    population.append(chromo)
        
        
best = 1000000000000000000
best_seq = 0
for i in range(epochs):
    new_population = []
    print("reproduction")
    for j in range(offspring_no):
        x = np.random.randint(len(population))
        y = np.random.randint(len(population)) 
        child1, child2 = crossover(population[x],population[y])
        new_population.append(child1)
        new_population.append(child2)
    population += new_population
    print("mutation")
    for individual in population:
        new_population.append(mutation(individual))
    print("scoring")
    population.sort(key=lambda x:cost_function(x))
    population = population[:population_no]
    print(cost_function(population[0]))
    if best > cost_function(population[0]):
        best = cost_function(population[0])
        best_seq = population[0]
print(best)