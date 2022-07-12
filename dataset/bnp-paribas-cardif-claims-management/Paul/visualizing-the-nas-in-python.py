
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load data
train=pd.read_csv('../input/train.csv')
train.drop(['ID', 'target'], axis=1, inplace=True)

# count NAs for each column
v = train.apply(lambda x: sum(x.isnull().values), axis=0)

# sorting NAs
v.sort_values(ascending=False, inplace=True)  
print(v)

# plotting
n = len(v)
x = np.arange(1, n+1)
plt.bar(x, v, 1)
plt.xticks(x + 1/2.0, v.index)
plt.xticks(rotation=90) # or use 'vertical'
plt.savefig('na_visualization.jpg')


# TODO: increase space between xticks, or make the label smaller...