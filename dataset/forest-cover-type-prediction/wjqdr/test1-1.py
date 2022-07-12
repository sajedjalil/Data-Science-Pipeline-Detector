import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

# Any files you write to the current directory get shown as outputs
train = train.drop('Id', axis = 1)

#print(train.head())
print(train.shape)
#print(train.dtypes)
#print(train.describe())
#print(train.groupby('Cover_Type'))
#print(train.groupby('Cover_Type').size())

rem = []

for c in train.columns:
    if train[c].std() == 0:
        rem.append(c)

train = train.drop(rem, axis = 1)

print(rem)
print(train.shape)
























