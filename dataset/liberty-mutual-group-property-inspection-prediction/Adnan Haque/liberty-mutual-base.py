# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from time import clock
import matplotlib.pyplot as plt

print (plt.style.available)

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

print ("Concatenating train and test sets.")
df = pd.concat([train,test])

print ("Creating scatter matrix visualization.")
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.savefig("scatter-matrix.png")

# Write summaries of the train and test sets to the log
print('\nSummary of train dataset:\n')
print(train.describe())
print('\nSummary of test dataset:\n')
print(test.describe())
