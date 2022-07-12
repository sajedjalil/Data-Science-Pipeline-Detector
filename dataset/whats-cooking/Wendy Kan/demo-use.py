
import pandas as pd

# Reading the data
train = pd.read_json('../input/train.json')

print(train[train.cuisine=='chinese'])