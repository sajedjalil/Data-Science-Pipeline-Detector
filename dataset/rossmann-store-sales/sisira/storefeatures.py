""" GjC 2015 kaggle: Rossmann """
""" Simple script to extract promo interval as features """

import pandas as pd
import numpy as np

infile = '../input/store.csv'
outfile = 'stores_feat.csv'
stores = pd.read_csv(infile, dtype=object)

print(stores.head(), '\n')

