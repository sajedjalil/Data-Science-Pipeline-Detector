import pandas as pd
import os

os.system("ls ../input")

train = pd.read_csv("../input/train.csv")
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

test = pd.read_csv("../input/test.csv")
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

# Allow very wide arrays to be printed in full
pd.options.display.max_columns = None

print("Head of training set:")
print(train.head())

print("pandas.describe():")
print(train.describe())

