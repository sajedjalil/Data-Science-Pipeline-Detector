import pandas as pd

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
store = pd.read_csv("../input/store.csv")

print(train.describe())
print(store.describe())
print(store.head())
