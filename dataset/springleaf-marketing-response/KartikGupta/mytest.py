# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd

# The competition datafiles are in the directory ../input
# List the files we have available to work with
print("> ls ../input")
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Read train data file:
train = pd.read_csv("../input/train.csv")
part1 = train[train['target']==1]
part2 = train[train['target']==0]

# Write summaries of the train and test sets to the log
print('\nSummary of train 1 dataset:\n')
print(part1.describe())
print("\npart 2\n")
print(part2.describe())