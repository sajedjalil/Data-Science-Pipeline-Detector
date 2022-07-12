# -*- coding: utf-8 -*-
"""
@author: Faron
"""
import pandas as pd
import numpy as np

DATA_DIR = "../input"

ID_COLUMN = 'Id'
TARGET_COLUMN = 'Response'
nonDupesDate = ["Id", "L0_S0_D1","L0_S1_D26","L0_S10_D216","L0_S10_D221","L0_S10_D231","L0_S11_D280","L0_S12_D331","L0_S13_D355","L0_S14_D360","L0_S15_D395","L0_S16_D423","L0_S17_D432","L0_S18_D437","L0_S19_D454","L0_S2_D34","L0_S20_D462","L0_S21_D469","L0_S21_D474","L0_S21_D484","L0_S22_D543","L0_S22_D548","L0_S22_D553","L0_S22_D558","L0_S23_D617","L0_S3_D70","L0_S4_D106","L0_S4_D111","L0_S5_D115","L0_S6_D120","L0_S7_D137","L0_S8_D145","L0_S9_D152","L0_S9_D157","L0_S9_D167","L0_S10_D216","L0_S10_D221","L0_S10_D231","L0_S11_D280","L0_S12_D331","L0_S13_D355","L0_S14_D360","L0_S15_D395","L0_S16_D423","L0_S17_D432","L0_S18_D437","L0_S19_D454","L0_S20_D462","L0_S21_D469","L0_S21_D474","L0_S21_D484","L0_S22_D543","L0_S22_D548","L0_S22_D553","L0_S22_D558","L0_S23_D617","L1_S24_D677","L1_S24_D697","L1_S24_D702","L1_S24_D772","L1_S24_D801","L1_S24_D804","L1_S24_D807","L1_S24_D813","L1_S24_D818","L1_S24_D909","L1_S24_D999","L1_S24_D1018","L1_S24_D1062","L1_S24_D1116","L1_S24_D1135","L1_S24_D1155","L1_S24_D1158","L1_S24_D1163","L1_S24_D1168","L1_S24_D1171","L1_S24_D1178","L1_S24_D1186","L1_S24_D1277","L1_S24_D1368","L1_S24_D1413","L1_S24_D1457","L1_S24_D1511","L1_S24_D1522","L1_S24_D1536","L1_S24_D1558","L1_S24_D1562","L1_S24_D1566","L1_S24_D1568","L1_S24_D1570","L1_S24_D1576","L1_S24_D1583","L1_S24_D1674","L1_S24_D1765","L1_S24_D1770","L1_S24_D1809","L1_S24_D1826","L1_S25_D1854","L1_S25_D1867","L1_S25_D1883","L1_S25_D1887","L1_S25_D1891","L1_S25_D1898","L1_S25_D1902","L1_S25_D1980","L1_S25_D2058","L1_S25_D2098","L1_S25_D2138","L1_S25_D2180","L1_S25_D2206","L1_S25_D2230","L1_S25_D2238","L1_S25_D2240","L1_S25_D2242","L1_S25_D2248","L1_S25_D2251","L1_S25_D2329","L1_S25_D2406","L1_S25_D2430","L1_S25_D2445","L1_S25_D2471","L1_S25_D2497","L1_S25_D2505","L1_S25_D2507","L1_S25_D2509","L1_S25_D2515","L1_S25_D2518","L1_S25_D2596","L1_S25_D2674","L1_S25_D2713","L1_S25_D2728","L1_S25_D2754","L1_S25_D2780","L1_S25_D2788","L1_S25_D2790","L1_S25_D2792","L1_S25_D2798","L1_S25_D2801","L1_S25_D2879","L1_S25_D2957","L1_S25_D2996","L1_S25_D3011","L2_S26_D3037","L2_S26_D3081","L2_S27_D3130","L2_S28_D3223","L3_S29_D3316","L3_S29_D3474","L3_S30_D3496","L3_S30_D3506","L3_S30_D3566","L3_S30_D3726","L3_S31_D3836","L3_S32_D3852","L3_S33_D3856","L3_S34_D3875","L3_S35_D3886","L3_S35_D3895","L3_S35_D3900","L3_S36_D3919","L3_S36_D3928","L3_S37_D3942","L3_S38_D3953","L3_S39_D3966","L3_S40_D3981","L3_S40_D3985","L3_S41_D3997","L3_S42_D4029","L3_S42_D4045","L3_S43_D4062","L3_S43_D4082","L3_S44_D4101","L3_S45_D4125","L3_S46_D4135","L3_S47_D4140","L3_S48_D4194","L3_S49_D4208","L3_S49_D4218","L3_S50_D4242","L3_S51_D4255"]

SEED = 0
CHUNKSIZE = 50000
NROWS = 1183748

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
for tr, te in zip(pd.read_csv(TRAIN_DATE, chunksize=CHUNKSIZE, usecols = nonDupesDate), pd.read_csv(TEST_DATE, usecols = nonDupesDate, chunksize=CHUNKSIZE)):
    print(nrows)
    feats = np.setdiff1d(tr.columns, [ID_COLUMN])

    stime_tr = tr[feats].min(axis=1).values
    stime_te = te[feats].min(axis=1).values

    train.loc[train.Id.isin(tr.Id), 'StartTime'] = stime_tr
    test.loc[test.Id.isin(te.Id), 'StartTime'] = stime_te

    nrows += CHUNKSIZE
    if nrows >= NROWS:
        break


ntrain = train.shape[0]
train_test = pd.concat((train, test)).reset_index(drop=True).reset_index(drop=False)

train_test['f_1'] = train_test[ID_COLUMN].diff().fillna(9999999).astype(int)
train_test['f_2'] = train_test[ID_COLUMN].iloc[::-1].diff().fillna(9999999).astype(int)

train_test = train_test.sort_values(by=['StartTime', 'Id'], ascending=True)

train_test['f_3'] = train_test[ID_COLUMN].diff().fillna(9999999).astype(int)
train_test['f_4'] = train_test[ID_COLUMN].iloc[::-1].diff().fillna(9999999).astype(int)

train_test['Magic'] = 1 + 2 * (train_test['f_3'] > 1) + 1 * (train_test['f_4'] < -1)

train_test = train_test.sort_values(by=['index']).drop(['index', 'Response'], axis=1)
print(train_test[:ntrain].head())
print(train_test[ntrain:].head())
train_test[:ntrain].to_csv("Farons_train_features6.csv.gz",index=False, compression="gzip")
train_test[ntrain:].to_csv("Farons_test_features6.csv.gz", index=False, compression="gzip")
print("Done.....")




