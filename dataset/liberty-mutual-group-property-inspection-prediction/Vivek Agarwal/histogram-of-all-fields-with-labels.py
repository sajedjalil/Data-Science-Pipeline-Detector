# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The competition datafiles are in the directory ../input
# Read competition data files:
test = pd.read_csv("../input/test.csv")

# preprocessing columns
# factorizing column values
test['T1_V4'] = pd.factorize(test['T1_V4'])[0]
test['T1_V5'] = pd.factorize(test['T1_V5'])[0]
# simple yes/no
test['T1_V6'] = pd.factorize(test['T1_V6'])[0]
test['T1_V7'] = pd.factorize(test['T1_V7'])[0]
test['T1_V8'] = pd.factorize(test['T1_V8'])[0]
test['T1_V9'] = pd.factorize(test['T1_V9'])[0]
test['T1_V11'] = pd.factorize(test['T1_V11'])[0]
test['T1_V12'] = pd.factorize(test['T1_V12'])[0]
test['T1_V15'] = pd.factorize(test['T1_V15'])[0]
test['T1_V16'] = pd.factorize(test['T1_V16'])[0]
test['T1_V17'] = pd.factorize(test['T1_V17'])[0]

test['T2_V3'] = pd.factorize(test['T2_V3'])[0]
test['T2_V5'] = pd.factorize(test['T2_V5'])[0]
test['T2_V11'] = pd.factorize(test['T2_V11'])[0]
test['T2_V12'] = pd.factorize(test['T2_V12'])[0]
test['T2_V13'] = pd.factorize(test['T2_V13'])[0]


fig = plt.figure(figsize=(15, 12))

# loop over all vars (total: 34)
for i in range(1, test.shape[1]):
    plt.subplot(6, 6, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(test.columns.values[i])
    # f.axes.set_ylim([0, test.shape[0]])

    vals = np.size(test.iloc[:, i].unique())
    if vals < 10:
        bins = vals
    else:
        vals = 10

    plt.hist(test.iloc[:, i], bins=30, color='#3F5D7D')

plt.tight_layout()

plt.savefig("histogram-distribution.png")