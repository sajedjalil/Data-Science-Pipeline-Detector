#!/usr/bin/env python3

import os

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

DATA_ROOT = '../input/'


def main():
    n_fold = 5
    depths = pd.read_csv(os.path.join(DATA_ROOT, 'depths.csv'))
    depths.sort_values('z', inplace=True)
    depths.drop('z', axis=1, inplace=True)
    depths['fold'] = (list(range(n_fold))*depths.shape[0])[:depths.shape[0]]
    print(depths.head())
    # depths.to_csv(os.path.join(DATA_ROOT, 'folds.csv'), index=False)


if __name__ == '__main__':
    main()