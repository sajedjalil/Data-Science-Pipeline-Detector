# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv", index_col = 0)
test  = pd.read_csv("../input/test.csv", index_col = 0)

# Write summaries of the train and test sets to the log
print('\nSummary of train dataset:\n')
print(train.describe())
print('\nSummary of test dataset:\n')
print(test.describe())

print('\train dataset:\n')
print(train.head())
print('\ntest dataset:\n')
print(test.head())