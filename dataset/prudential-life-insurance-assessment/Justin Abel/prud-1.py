import pandas as pd 
import numpy as np
from sklearn import svm
import  scipy.stats as stats
import matplotlib.pyplot as plt

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")

n = 10000 # Number of samples to use for training

train_rows = np.random.choice(train.index.values, n)
train_sample = train.ix[train_rows]
train_sample = train_sample.sort_index()

start_col = 4
end_col = 126

# Select the features data
X_raw = train_sample.ix[:,start_col:end_col]

# Select the labels
y = train_sample.ix[:,'Response']

# Normalise X
normalise = lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x))   
X = X_raw.apply(normalise)

# Replace invalid values with zeroes
for column in X:
    nans = np.isnan(X.ix[:,column])
    col_mean = X.ix[:,column].mean()
    if np.isnan(col_mean):
        col_mean = 0
    X.ix[nans, column] = col_mean
    
print(X)
    
# Select the test sample
test_sample_raw = train.ix[10001:10010,start_col:end_col]

# Normalise test_sample
normalise = lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x))   
test_sample = test_sample_raw.apply(normalise)

# Replace invalid values with zeroes
for column in test_sample:
    nans = np.isnan(test_sample.ix[:,column])
    col_mean = test_sample.ix[:,column].mean()
    if np.isnan(col_mean):
        col_mean = 0
    test_sample.ix[nans, column] = col_mean

# Train the model
clf = svm.SVC()
fit = clf.fit(X, y)
print(fit)

# Use model to predict labels for the test sample
result = clf.predict(test_sample)
print(result)
print(train.ix[10001:10010,'Response'])