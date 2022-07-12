#!/usr/bin/env python3

import os

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

DATA_ROOT = '../input/'


def main():
    train_ids = pd.read_csv(os.path.join(DATA_ROOT, 'train.csv')).id

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    folds = [[train_ids[idx] for idx in idxs[1]]
             for idxs in kfold.split(train_ids)]

    # np.save(os.path.join(DATA_ROOT, 'folds.npy'), folds)


if __name__ == '__main__':
    main()
