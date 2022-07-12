import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

# input files
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# drop some columns
train.drop("Hazard", axis=1, inplace=True)
train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)

# convert pandas dataframe to numpy array
train_s = train
test_s = test
train_s = np.array(train_s)
test_s = np.array(test_s)

# convert features to numeric values
for i in range(train_s.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_s[:,i]) + list(test_s[:,i]))
    train_s[:,i] = lbl.transform(train_s[:,i])
    test_s[:,i] = lbl.transform(test_s[:,i])

n_bins = 60
fig = plt.figure(figsize=(10,80))

# make subplots showing train/test feature distributions side by side
for i in range (1, train_s.shape[1]*2+1):
    if i % 2 != 0:
        plt.subplot(32, 2, i)
        plt.hist(train_s[:,(i-1)/2], bins=n_bins)
        j = i/2+1
        plt.xlabel("train feature #%d" % j)
        plt.ylabel("counts")
        fig.tight_layout()
    elif i % 2 == 0:
        plt.subplot(32, 2, i)
        plt.hist(test_s[:,((i-1)/2)-1/2], bins=n_bins)
        j = i/2
        plt.xlabel("test feature #%d" % j)
        plt.ylabel("counts")
        fig.tight_layout()
plt.gcf().savefig("train-test_feature_dist.png")
