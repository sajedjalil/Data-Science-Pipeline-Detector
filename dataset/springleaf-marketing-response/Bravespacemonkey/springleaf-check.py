# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import theano
import theano.tensor as T
x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = theano.function([x], s)
print(logistic([[0, 1], [-1, -2]]))


# The competition datafiles are in the directory ../input
# List the files we have available to work with
print("> ls ../input")
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Read train data file:
train = pd.read_csv("../input/train.csv")

# Write summaries of the train and test sets to the log
print('\nSummary of train dataset:\n')
#print(train.describe())
print(train)