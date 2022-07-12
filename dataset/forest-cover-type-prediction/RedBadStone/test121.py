import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
print("Hello,world")

# Any files you write to the current directory get shown as outputs