# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd

# The competition datafiles are in the directory ../input
# List the files we have available to work with
print("> ls ../input")
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Read competition data files:
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")

# Write summaries of the train and test sets to the log
print('\nSummary of train dataset:\n')
print(train.describe())
print('\nSummary of test dataset:\n')
print(test.describe())

