# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")

# preprocessing columns
# factorizing column values
train['T1_V4'] = pd.factorize(train['T1_V4'])[0]
train['T1_V5'] = pd.factorize(train['T1_V5'])[0]
# simple yes/no
train['T1_V6'] = pd.factorize(train['T1_V6'])[0]
train['T1_V7'] = pd.factorize(train['T1_V7'])[0]
train['T1_V8'] = pd.factorize(train['T1_V8'])[0]
train['T1_V9'] = pd.factorize(train['T1_V9'])[0]
train['T1_V11'] = pd.factorize(train['T1_V11'])[0]
train['T1_V12'] = pd.factorize(train['T1_V12'])[0]
train['T1_V15'] = pd.factorize(train['T1_V15'])[0]
train['T1_V16'] = pd.factorize(train['T1_V16'])[0]
train['T1_V17'] = pd.factorize(train['T1_V17'])[0]

train['T2_V3'] = pd.factorize(train['T2_V3'])[0]
train['T2_V5'] = pd.factorize(train['T2_V5'])[0]
train['T2_V11'] = pd.factorize(train['T2_V11'])[0]
train['T2_V12'] = pd.factorize(train['T2_V12'])[0]
train['T2_V13'] = pd.factorize(train['T2_V13'])[0]

train['T2_V2'] = np.log(train['T2_V2'])
train['T2_V4'] = np.log(train['T2_V4'])
#train['Hazard'] = np.log(train['Hazard'])

print(np.sort(train['Hazard'].unique()))
print(train['Hazard'].value_counts())

fig = plt.figure(figsize=(15, 12))

# loop over all vars (total: 34)
j = 0
for i in train:
    j+=1
    plt.subplot(6, 6, j)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    plt.title(i)
    # f.axes.set_ylim([0, train.shape[0]])

    vals = np.size(train.loc[:, i].unique())
    if vals < 10:
        bins = vals
    else:
        vals = 10

    plt.hist(train.loc[:, i], bins=30, color='#3F5D7D')

plt.tight_layout()

plt.savefig("histogram-distribution.png")