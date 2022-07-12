# Santander - Association Rules with Orange3-Associate

import numpy as np
import pandas as pd
from orangecontrib.associate.fpgrowth import * # Orange3-Associate module

# Use only data from the last month for users in testset
test = pd.read_csv('../input/test_ver2.csv')

train_lastmonth = pd.DataFrame()

trainIterator = pd.read_csv('../input/train_ver2.csv', chunksize=1000000) # filter train in chunks
for chunk in trainIterator:     
    train_lastmonth = train_lastmonth.append(chunk[(chunk.fecha_dato == '2016-05-28') & 
                                                   (chunk.ncodpers.isin(test.ncodpers))])

# Convert table to Series with list of products
cols = train_lastmonth.columns[24:]
 
prod_theyhave = pd.Series([cols[row[24:] == 1].tolist() for row in train_lastmonth.values], 
                          index=train_lastmonth.ncodpers)

prod_donthave = pd.Series([cols[row[24:] == 0].tolist() for row in train_lastmonth.values], 
                          index = train_lastmonth.ncodpers)

# Generate basic probabilities for products
prod_rank = (train_lastmonth.iloc[:,24:].mean()).sort_values()[::-1]

# Generate Association Rules
itemsets = dict(frequent_itemsets(prod_theyhave.tolist(), .005))
rules = list(association_rules(itemsets, .01))
rules = [rule for rule in rules if len(rule[1]) == 1] # only keep rules with one outcome variable

# Assign basic probabilities to products customers dont have yet
prod_prob = train_lastmonth.iloc[:,24:].copy()
prod_prob.index = train_lastmonth.ncodpers

for prod in cols: # Assign basic probabilities
    prod_prob.loc[:,prod] = prod_rank[prod] 
for customer in prod_prob.index: # Prob = 0 if customer has it already
    prod_prob.loc[customer, prod_theyhave[customer]] = 0

# Update Probabilities with Association Rules
for customer in prod_prob.index:
    for prod in prod_donthave[customer]:
        for rule in rules:
            if prod == list(rule[1])[0] and rule[0] <= set(prod_theyhave[customer]): # check if rule applies
                if rule[3] > prod_prob.loc[customer,prod]: # update only if rule indicates new higher probability
                    prod_prob.loc[customer,prod] = rule[3]
                    
# Generate submission with 7 most probable products
reco = [list(prod_prob.loc[customer,:].sort_values()[:-8:-1].index) for customer in prod_prob.index]

submission = pd.DataFrame({'ncodpers': prod_prob.index, 
                           'added_products': [' '.join(prod) for prod in reco]})

submission.to_csv('submission.csv', index=False)