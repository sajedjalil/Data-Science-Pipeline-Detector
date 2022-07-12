import pandas as pd 

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")

print(train.shape)
print(train.head(n=5))
print(test.shape)
print(test.head(n=5))
