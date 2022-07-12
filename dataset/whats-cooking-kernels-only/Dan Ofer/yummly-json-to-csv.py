import os
print(os.listdir("../input"))
import pandas as pd

train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')

train.to_csv("yummly_train.csv",index=False)
test.to_csv("yummly_test.csv",index=False)