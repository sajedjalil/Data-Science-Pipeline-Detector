import pandas as pd
import numpy as np

train = pd.read_csv('../input/train.csv')
print(train.info())

remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)

train.drop(remove, axis=1, inplace=True)
print(train.replace(0, np.nan).to_sparse().info())