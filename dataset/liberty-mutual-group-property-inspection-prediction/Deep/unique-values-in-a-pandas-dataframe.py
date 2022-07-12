import pandas as pd
import numpy as np

df = pd.DataFrame({'b':[2,8,4,5], 'c': [3,3,0,9], 'd': [3,5,8,9]}, index=[1,2,3,4])

print(df)

for i in range(0,4):
    u = pd.unique(df.iloc[i])
    print('The unique values in this row are: ' + np.array_str(u))
    print('The number of unique values in row: ' + str(i) + ' is: ' + str(len(u)))
