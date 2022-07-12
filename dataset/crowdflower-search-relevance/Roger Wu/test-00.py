
import pandas as pd

# Use Pandas to read in the training and test data
train = pd.read_csv("../input/train.csv").fillna("")
test  = pd.read_csv("../input/test.csv").fillna("")

# Print a sample of the training data
print(train.head())

# Now it's yours to take from here!
