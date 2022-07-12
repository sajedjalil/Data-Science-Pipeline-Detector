# -*- coding: utf-8 -*-
"""
@author: Faron
"""
import pandas as pd
import numpy as np
import xgboost as xgb

DATA_DIR = "../input"

ID_COLUMN = 'Id'
TARGET_COLUMN = 'Response'

SEED = 0
CHUNKSIZE = 50000
NROWS = 250000

TRAIN_NUMERIC = "{0}/train_numeric.csv".format(DATA_DIR)
TRAIN_DATE = "{0}/train_date.csv".format(DATA_DIR)

TEST_NUMERIC = "{0}/test_numeric.csv".format(DATA_DIR)
TEST_DATE = "{0}/test_date.csv".format(DATA_DIR)

FILENAME = "etimelhoods"

train = pd.read_csv(TRAIN_NUMERIC, usecols=[ID_COLUMN, TARGET_COLUMN], nrows=NROWS)
test = pd.read_csv(TEST_NUMERIC, usecols=[ID_COLUMN], nrows=NROWS)

train["StartTime"] = -1
test["StartTime"] = -1

nrows = 0
for tr, te in zip(pd.read_csv(TRAIN_DATE, chunksize=CHUNKSIZE), pd.read_csv(TEST_DATE, chunksize=CHUNKSIZE)):
    feats = np.setdiff1d(tr.columns, [ID_COLUMN])
    print(feats)
    stime_tr = tr[feats].min(axis=1).values
    stime_te = te[feats].min(axis=1).values

    train.loc[train.Id.isin(tr.Id), 'StartTime'] = stime_tr
    test.loc[test.Id.isin(te.Id), 'StartTime'] = stime_te

    nrows += CHUNKSIZE
    if nrows >= NROWS:
        break

