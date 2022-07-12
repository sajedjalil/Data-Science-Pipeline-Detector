
# Merry Christmas Y'all (Python Version)

# This is a python version of Merry Christmas Y'all, originally written by Ben Gorman in R. 
# The python version is written by Wenjian Hu

#======================================================================================================
# How it works

# First, some much need naming conventions (seriously guys, can we agree to use these names throughout the competition?)

# gift: a specific, physical item (e.g. train_11)
# toy: a type of gift (e.g. horse, ball, etc.)
# bag: a physical bag
# baggiftsmap: a mapping of gifts to a bag (Bag_7, horse_20, horse_21, train_11)
# bagtoys: generic combinations of toys (2 horses, 1 train)
# bagtoysmap: a mapping of toys to bags (Bag_7, 2 horses, 1 train)
# bag score: this is the actual score the bag receives (i.e. the weight of the bag or 0 if weight > 50)

#======================================================================================================
# Load packages 

import numpy as np
import pandas as pd
from pulp import *  # to install, simply use: pip install pulp 

#======================================================================================================
# Helper methods for sampling the weight distributions

dispatch = {
    "horse"  : lambda k : np.maximum(0, np.random.normal(5,2, size=k)),
    "ball"   : lambda k : np.maximum(0, 1 + np.random.normal(1,0.3, size=k)),
    "bike"   : lambda k : np.maximum(0, np.random.normal(20,10, size=k)),
    "train"  : lambda k : np.maximum(0, np.random.normal(10,5, size=k)),
    "coal"   : lambda k : 47 * np.random.beta(0.5,0.5, size=k),
    "book"   : lambda k : np.random.chisquare(2, size=k),
    "doll"   : lambda k : np.random.gamma(5,1, size=k),
    "blocks" : lambda k : np.random.triangular(5,10,20, size=k),
    "gloves" : lambda k : np.random.rand(k) + (np.random.rand(k)<0.3)*3
}
#======================================================================================================
# Load the data

#--------------------------------------------------
# load gifts
gifts = pd.read_csv("../input/gifts.csv")
gifts['Toy'] = gifts['GiftId'].apply(lambda x: x.split("_")[0])

#--------------------------------------------------
# toys
toys = gifts.groupby(['Toy']).agg(['count']).reset_index()
toys.columns = ['Toy', 'Stock']
toys['ToyPlural']  = np.asarray(["Balls", "Bikes", "Blocks", "Books", "Coals", "Dolls", "Gloves", "Horses", "Trains"])

# Get the expected weight of each toy using simulation
k = 200000
for toy in dispatch:
  Weights = dispatch[toy](k)
  toys.loc[toys['Toy']==toy, 'EWeight'] = np.mean(Weights)

print('Get a general idea of toy weights:')
print(toys)
#======================================================================================================
# Part1

# The goal here is to generate a set of bagtoys that we might use to fill a bag. 
# For example, a bagtoys might be like (2 horses, 1 train) or (5 balls). We  will use a bagtoys as a blueprint for filling 1 or more bags with toys
# Additionally, for a given bagtoys we'll calculate its expected score for filling 1 bag

#--------------------------------------------------
# Make some bagtoys
v0 = np.concatenate((np.repeat(0, 3), np.repeat(1, 5), np.repeat(2, 4), np.repeat(3, 3), np.repeat(4, 2)) )
v1 = np.concatenate((np.repeat(0, 10), np.repeat(1, 5), np.repeat(2, 2), np.repeat(3, 1), np.repeat(4, 1)) )
v2 = np.concatenate((np.repeat(0, 13), np.repeat(1, 5), np.repeat(2, 1)) )
k = 50000
seeds = [123, 234, 345, 456, 567, 678, 789, 890, 901]
bagtoys = pd.DataFrame()
np.random.seed(seeds[0])
bagtoys['Balls']=np.random.choice(v0, size=k, replace=True) 
np.random.seed(seeds[1])
bagtoys['Bikes']=np.random.choice(v2, size=k, replace=True) 
np.random.seed(seeds[2])
bagtoys['Blocks']=np.random.choice(v1, size=k, replace=True) 
np.random.seed(seeds[3])
bagtoys['Books']=np.random.choice(v0, size=k, replace=True) 
np.random.seed(seeds[4])
bagtoys['Coals']=np.random.choice(v2, size=k, replace=True) 
np.random.seed(seeds[5])
bagtoys['Dolls']=np.random.choice(v1, size=k, replace=True) 
np.random.seed(seeds[6])
bagtoys['Gloves']=np.random.choice(v0, size=k, replace=True) 
np.random.seed(seeds[7])
bagtoys['Horses']=np.random.choice(v0, size=k, replace=True) 
np.random.seed(seeds[8])
bagtoys['Trains']=np.random.choice(v1, size=k, replace=True) 


# Exclude duplicates and bagtoys that violate the >= 3 constraint
bagtoys = bagtoys.drop_duplicates()
bagtoys = bagtoys.loc[ bagtoys['Balls'] + bagtoys['Bikes'] + bagtoys['Blocks'] + bagtoys['Books'] + bagtoys['Coals'] + bagtoys['Dolls'] 
                                            + bagtoys['Gloves'] + bagtoys['Horses'] + bagtoys['Trains'] >= 3 ,:].reset_index(drop=True)

#--------------------------------------------------
# Get the expected score for each bagtoy if it were to fill 1 bag
# EScore = Prob(weight <= 50) * E[weight | weight <= 50]
# We'll use simulation to calculate Prob(weight <= 50)
# (Tried to do this directly but those truncated distributions make for some nasty multiple integrals)

def expected_bagtoys_score(trials=1000, ballsN=0, bikesN=0, blocksN=0, booksN=0, coalsN=0, dollsN=0, glovesN=0, horsesN=0, trainsN=0):
  # Calculate the expected score using simulation

  balls = dispatch['ball'](trials * ballsN)
  bikes = dispatch['bike'](trials * bikesN)
  blocks = dispatch['blocks'](trials * blocksN)
  books = dispatch['book'](trials * booksN)
  coals = dispatch['coal'](trials * coalsN)
  dolls = dispatch['doll'](trials * dollsN)
  gloves = dispatch['gloves'](trials * glovesN)
  horses = dispatch['horse'](trials * horsesN)
  trains = dispatch['train'](trials * trainsN)

  Weight = np.concatenate((balls, bikes, blocks, books, coals, dolls, gloves, horses, trains))
    
  TrialId = np.concatenate( (
      np.repeat(np.asarray(range(trials))+1, ballsN),
      np.repeat(np.asarray(range(trials))+1, bikesN),
      np.repeat(np.asarray(range(trials))+1, blocksN),
      np.repeat(np.asarray(range(trials))+1, booksN),
      np.repeat(np.asarray(range(trials))+1, coalsN),
      np.repeat(np.asarray(range(trials))+1, dollsN),
      np.repeat(np.asarray(range(trials))+1, glovesN),
      np.repeat(np.asarray(range(trials))+1, horsesN),
      np.repeat(np.asarray(range(trials))+1, trainsN)
  ) )
  # Insert the results into a dataframe
  dt = pd.DataFrame()
  dt['TrialId'] = TrialId
  dt['Weight'] = Weight
  # Aggregate
  trials = dt.groupby(['TrialId']).agg(['sum']).reset_index()
  trials.columns = ['TrialId', 'Weight']
  trials['Score'] = trials['Weight']
  trials.loc[trials.Weight > 50, 'Score'] = 0

  return(trials['Score'].mean())

# Get the expected score of each bagtoys
for i in range(bagtoys.shape[0]):
  # Print the progress
  if((i+1) % 1000 == 0): print("Iteration: {}".format(i+1))
  
  bagtoys.ix[i, 'EScore'] = expected_bagtoys_score(
    trials=10000, ballsN=bagtoys.ix[i, 'Balls'], bikesN=bagtoys.ix[i, 'Bikes'], blocksN=bagtoys.ix[i, 'Blocks'], booksN=bagtoys.ix[i, 'Books'],
    coalsN=bagtoys.ix[i, 'Coals'], dollsN=bagtoys.ix[i, 'Dolls'], glovesN=bagtoys.ix[i, 'Gloves'], horsesN=bagtoys.ix[i, 'Horses'], trainsN=bagtoys.ix[i, 'Trains']
  )

# Sort, rank, and cleanup
bagtoys = bagtoys.sort_values(by='EScore', axis=0, ascending=False).reset_index(drop=True)
bagtoys['BagToysId'] = bagtoys.reset_index()['index']
cols = ["BagToysId", "EScore", "Balls", "Bikes", "Blocks", "Books", "Coals", "Dolls", "Gloves", "Horses", "Trains"]
bagtoys = bagtoys.ix[:, cols]

#======================================================================================================
# Part 2

# Now the goal is to determine the optimal combination of bagtoys to fill the bags with. 
# For example, we might choose to fill 300 bags using bagtoys_7 = (2 balls, 1 horse, 1 block), 250 bags with  bagtoys_8 = (5 balls) and so on.
# Notice, we're looking for the best linear combination of bagtoys, subject to some linear constraints.
# We can solve this using linear programming https://en.wikipedia.org/wiki/Linear_programming

def pack_lp(bagtoys):
  # define the problem
  prob = LpProblem("The Santa Uncertain Bags Problem", LpMaximize)

  # collect coeffients for the linear programming problem: A*x <= b
  cols = ["Balls", "Bikes", "Blocks", "Books", "Coals", "Dolls", "Gloves", "Horses", "Trains"]
  A = (bagtoys.loc[:, cols].as_matrix()).T
  b = toys['Stock'].as_matrix()

  # define variables
  vals_name=[str(i) for i in range(A.shape[1])]
  variables = LpVariable.dicts("x", vals_name, 0, None, LpInteger)

  # Objective (we want to maximize) c*x
  c = bagtoys['EScore'].as_matrix() 
  prob += lpSum([c[i] * variables[vals_name[i]] for i in range(A.shape[1])]), "objective"

  # Constraints (9 toy constraints (restricted by stocks) + 1 global constraint (restricted by 1000 bags))
  for i in range(A.shape[0]):
    prob += lpSum([A[i][j] * variables[vals_name[j]] for j in range(A.shape[1])]) <= b[i], ""
  prob += lpSum([variables[vals_name[i]] for i in range(A.shape[1])]) <= 1000, ""
  
  # Solve it
  prob.solve()
  print ("Status:", LpStatus[prob.status])

  # get variable values and reordered them properly
  vals = {}
  for v in prob.variables():
    vals[v.name]=v.varValue
  vals = [vals['x_'+str(i)] for i in range(A.shape[1])]

  # Return a dict with the score and the coeffs
  return({'Score': value(prob.objective), 'Bags': vals})

bagtoys = bagtoys.iloc[:6000, :]  # Greedily pick out the 6K bagtoys with the highest expected score
best = pack_lp(bagtoys)

print('the expected score:')
print(best['Score'])  

# Get the coefficients to determine how many of each bagtoys to use
bagtoys['Bags'] = best['Bags']
bagtoysUsed = bagtoys.loc[bagtoys.Bags > 0, :].sort_values(by='Bags', axis=0, ascending=False).reset_index(drop=True)
bagtoysUsed['Bags'] = bagtoysUsed['Bags'].astype(int)

print('This is the solution:')
print(bagtoysUsed)

#======================================================================================================

# Clean up for submission
def baggifts_map(bagtoys):
  # Create a baggiftsmap (BagId, GiftId) from a bagtoys dataset
  # bagtoys should be in wide format with columns {BagToysId, Bags, Balls, Bikes, ...}
  # manipulate bagtoys to the proper format for submission

  ToyPlural = ["Balls", "Bikes", "Blocks", "Books", "Coals", "Dolls", "Gloves", "Horses", "Trains"]
  reqcols = ["BagToysId", "Bags"] + ToyPlural
  assert len(set(reqcols) - set(bagtoys.columns.values)) == 0
  
  dt = pd.melt(bagtoys, id_vars=["BagToysId", "Bags"], value_vars=ToyPlural, var_name='Toy', value_name='Gifts')
  dt = dt.loc[dt.Gifts>0, :]
  dt = dt.loc[np.repeat(dt.index.values, dt.Gifts), :].drop('Gifts', axis=1).reset_index(drop=True)
  dt['Id'] = dt.reset_index()['index'] + 1
  
  BagIdx = []
  for i in dt['Bags'].as_matrix():
    for j in range(i):
      BagIdx.append(j+1)
  dt = dt.loc[np.repeat(dt.index.values, dt.Bags), :].reset_index(drop=True)
  dt['BagIdx'] = np.asarray(BagIdx)
  dt = dt.drop('Bags', axis=1)
  
  dt['BagId'] = dt.groupby(['BagToysId', 'BagIdx']).grouper.group_info[0]
  Toy = ["ball", "bike", "blocks", "book", "coal", "doll", "gloves", "horse", "train"]
  for i in range(len(ToyPlural)):
    dt.loc[dt.Toy==ToyPlural[i], 'Toy'] = Toy[i] 
  
  dt_grouped = dt.groupby(['Toy'])
  for i in range(len(Toy)):
    if(dt.loc[dt.Toy==Toy[i],:].empty): continue
    dt.loc[dt.Toy==Toy[i],'ToyIdx'] = dt_grouped.get_group(Toy[i]).reset_index(drop=True).reset_index()['index'].as_matrix()

  dt['ToyIdx'] = dt['ToyIdx'].astype(int)
  dt['GiftId'] = dt['Toy'] + '_' + dt['ToyIdx'].apply(str)
  dt = dt.loc[:, ['BagId', 'GiftId', 'BagToysId', 'Toy']]
  
  return(dt)

# output store the proper format for submission
output = baggifts_map(bagtoysUsed)
output = output.loc[:, ['BagId', 'GiftId']].sort_values(by='BagId', axis=0).reset_index(drop=True)
output = output.groupby('BagId')['GiftId'].apply(lambda x: "%s" % ' '.join(x)).reset_index()['GiftId']
# rename column name
output = output.rename('Gifts')

# Finally
output.to_csv('sub.csv', index=False, header=True)