import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from IPython.display import display_markdown as mkdown # as print

def nl():
    print('\n')

for f in os.listdir('../input'):
    print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')
# #### It looks like we are given quite a few sets as an input! Let's take a look at each one, starting with train and test.

df_train = pd.read_csv('../input/train.csv', nrows=500000)
df_test = pd.read_csv('../input/test.csv', nrows=500000)

nl()
print('Size of training set: ' + str(df_train.shape))
print(' Size of testing set: ' + str(df_test.shape))

nl()
print('Columns in train: ' + str(df_train.columns.tolist()))
print(' Columns in test: ' + str(df_test.columns.tolist()))

nl()
print(df_train.describe())
# `Demanda_uni_equil` is the target value that we are trying to predict.
# 
# Let's take a look at the distribution:
target = df_train['Demanda_uni_equil'].tolist()

def label_plot(title, x, y):
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)

plt.hist(target, bins=200, color='blue')
label_plot('Distribution of target values', 'Demanda_uni_equil', 'Count')
plt.show()

print("Looks like we have some pretty big outliers, let's zoom in and try again")

print('Data with target values under 50: ' + str(round(len(df_train.loc[df_train['Demanda_uni_equil'] <= 50]) / 5000, 2)) + '%')

plt.hist(target, bins=50, color='blue', range=(0, 50))
label_plot('Distribution of target values under 50', 'Demanda_uni_equil', 'Count')
plt.show()

# From this distribution, we can see that some target values are much more common than others.
# 
# Let's find the mode of the target and make a naive submission using that!
from collections import Counter
print(Counter(target).most_common(10))
print('Our most common value is 2')

sub = pd.read_csv('../input/sample_submission.csv')
sub['Demanda_uni_equil'] = 2
sub.to_csv('mostcommon.csv', index=False)
# Interestingly, our script (0.96080) performs worse than submitting `6` as the predicted value. This could be for two reasons:
# 
# 1) Our values are incorrect since we have only read the first 500,000 values of the dataset and the set is not randomised.  
# 2) Due to the [evaluation metric](https://www.kaggle.com/c/grupo-bimbo-inventory-demand/details/evaluation) predicting 6 actually gives a lower overall logarithmic error.
# 
# We will begin by investigating the first possibility, and will look at whether the time-series has any effect on data.
pseudo_time = df_train.loc[df_train.Demanda_uni_equil < 20].index.tolist()
target = df_train.loc[df_train.Demanda_uni_equil < 20].Demanda_uni_equil.tolist()

plt.hist2d(pseudo_time, target, bins=[50,20])
label_plot('Histogram of target value over index', 'Index', 'Target')
plt.show()
# It does not look like the time-series has much effect on the data, except for that anomaly around 200k (we may take a closer look at another time)
# 
# To test out option 2, I created a script which evaluates the RMSLE on the training set to try and find the best value to submit, and it scored 0.82735:  
# https://www.kaggle.com/anokas/grupo-bimbo-inventory-demand/optimised-beat-the-benchmark
# 
# Now that we have found the best naive submission to make, we can go onto looking at the other columns!
# 
# We will begin by looking at the time column, semana (meaning week)
semana = df_train['Semana']
print(semana.value_counts())
print('\nIt looks like by sampling only the first 500,000 columns, we have only sampled from week 3.\nWe will have to take a larger portion of the dataset\n')

timing = pd.read_csv('../input/train.csv', usecols=['Semana','Demanda_uni_equil'])
print('Size: ' + str(timing.shape))

print(timing['Semana'].value_counts())
plt.hist(timing['Semana'].tolist(), bins=7, color='red')
label_plot('Distribution of weeks in training data', 'Semana', 'Frequency')
plt.show()

timing_test = pd.read_csv('../input/test.csv', usecols=['Semana'])
print(timing_test['Semana'].value_counts())
# We have a different set of weeks in the testing data for us to predict - meaning that this is likely a time series prediction problem for each of the product/client/location pairs in train and test sets.
# 
# Since this appears to be a time series prediction task, let's see if there are any trends in the target value over time.
timing = timing.sample(1000000)
timing = timing.loc[timing['Demanda_uni_equil'] < 15] # We only want to look at the most common values

plt.hist2d(timing['Semana'].tolist(), timing['Demanda_uni_equil'].tolist(), bins=[7, 15])
label_plot('Distribution of target value over time', 'Week', 'Target')
plt.show()