""" GjC 2015 kaggle: Rossmann """
""" Simple script to extract promo interval as features """

import pandas as pd
import numpy as np

infile = '../input/store.csv'
outfile = 'stores_feat.csv'
stores = pd.read_csv(infile, dtype=object)

print(stores.head(), '\n')

p_month = ['p_jan', 'p_feb', 'p_mar',\
            'p_apr', 'p_may', 'p_jun',\
            'p_jul', 'p_aug', 'p_sep',\
            'p_oct', 'p_nov', 'p_dec']
mon = ['Jan', 'Feb', 'Mar', 'Apr', 'May',\
        'Jun', 'Jul', 'Aug', 'Sept', 'Oct',\
        'Nov', 'Dec']

print('PromoInterval values:')
print(stores.PromoInterval.value_counts())
print('\nCould just as well be categorical variables, only 3 unique cases.\n')

for month in p_month:
    stores[month] = 0

stores.PromoInterval = stores.PromoInterval.fillna('Void')

for i in range(0, 12):
    stores[p_month[i]] = stores.PromoInterval.apply(lambda x: 1 if mon[i] in x.split(",") else 0)

stores.drop(['PromoInterval'], axis=1, inplace=True)

print(stores.head())
print('Saving output as: {}'.format(outfile))
stores.to_csv(outfile, index=False)
