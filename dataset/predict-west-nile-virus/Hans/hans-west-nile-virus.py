import pandas as pd
import os

os.system("ls ../input")

train = pd.read_csv("../input/train.csv")
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

print(train.head())

test = pd.read_csv("../input/test.csv")
print("Testing set has {0[0]} rows and {0[1]} columns".format(test.shape))

print(test.head())