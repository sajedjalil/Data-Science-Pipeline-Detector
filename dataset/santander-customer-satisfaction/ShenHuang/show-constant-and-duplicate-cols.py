import pandas as pd
import numpy as np

CSData = pd.read_csv("../input/train.csv")

# show constant columns
remove = []
for col in CSData.columns:
    if CSData[col].std() == 0:
        remove.append(col)
print(remove)

# show duplicated columns
remove = []
c = CSData.columns
for i in range(len(c)-1):
    v = CSData[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,CSData[c[j]].values):
            remove.append([c[i], c[j]])

print(remove)