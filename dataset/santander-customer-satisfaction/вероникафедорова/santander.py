import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
train = pd.read_csv("../input/train.csv")
train = pd.read_csv("../input/test.csv")
print(train.shape)

print(train.drop_duplicates(subset=train.columns[1:-1]).shape)
print(train.drop_duplicates(subset=train.columns[1:]).shape)
print(train.drop_duplicates(subset=train.columns[1:-1], keep=False).shape)
print(train.drop_duplicates(subset=train.columns[1:], keep=False).shape)
duplicate_ids = set(train['ID']).difference(set(train.drop_duplicates(subset=train.columns[1:-1], keep=False)['ID']))
duplicate_ids_2 = set(train['ID']).difference(set(train.drop_duplicates(subset=train.columns[1:], keep=False)['ID']))
print(len(duplicate_ids))
print(len(duplicate_ids_2))
to_drop = duplicate_ids.difference(duplicate_ids_2)
len(to_drop)
