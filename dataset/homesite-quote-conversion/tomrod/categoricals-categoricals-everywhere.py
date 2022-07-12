import pandas as pd

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train_nunique = {i: train[i].nunique() for i in train.columns}
print(train_nunique)
