import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("../input/train.csv", nrows=20000)

print('Remove columns...')
for col in ['VAR_0207', 'VAR_0213', 'VAR_0840', 'VAR_0847', 'VAR_1428', 'ID', 'target']:
    del train[col]

print('Keep only numeric...')
train = train.select_dtypes(include=['int64', 'float64'])

print('Remove...')
to_replace = [
    '-1', 999999997, 999999998, 999999999, 999999996, 999999994, 
    999994, 999999, -99999, 9999, 9996, 9990, 9998, 9999999
]
train.replace(to_replace=to_replace, value=np.nan)

print('Compute cardinalities...')
sets = train.apply(lambda x: [x.unique()])
cards = sets.apply(lambda x: len(x[0]))
plt.hist(cards, bins=100)
plt.savefig('cardinality.png')

dummys = []
cat = []
# Boolean
print('Boolean features...')
for col in sets[cards == 2].index:
    if train[col].value_counts().shape[0] == 1:
        dummys.append(col)
        # n_dummy += 1
        # Create dummy feature that tells is olumn is filled or not
        # train[col + '_dummy'] = train[col].notnull()
        # del train[col]
    else:
        cat.append(col)
        # n_cat += 1
        # train[col] = train[col].astype('category')

print('Low cardinality features...')
# For features with low cardinality compared to nrows (let's say < 20), 
# if numbers are 0,1,2,...,n we can assume that they stand for categories.
for col in sets[(cards > 2) & (cards <= 20)].index:
    s = sets[col][0]
    s = s[~np.isnan(s)]
    if set(s) == set(range(len(s))):
        cat.append(col)
print('- %s dummy to create' % len(dummys))
print(dummys)
print('')
print('- %s feature to cast to category' % len(cat))
print(cat)
print('')

print('Convert...')
# Create dummy feature that tells is olumn is filled or not
for col in dummys:
    train[col + '_dummy'] = train[col].notnull()
    del train[col]
for col in cat:
    train[col] = train[col].astype('category')

print('Done.')
    


