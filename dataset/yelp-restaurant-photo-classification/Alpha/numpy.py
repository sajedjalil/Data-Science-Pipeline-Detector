# Sample script naive benchmark that yields 0.609 public LB score WITHOUT any image information


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in data, files are assumed to be in the "../input/" directory.
train = pd.read_csv('../input/train.csv')
submit = pd.read_csv('../input/sample_submission.csv')

# convert numeric labels to binary matrix
def to_bool(s):
    return(pd.Series([1 if str(i) in str(s).split(' ') else 0 for i in range(9)]))
Y = train['labels'].apply(to_bool)

# get means proportion of each class
py = Y.mean()
plt.bar(Y.columns,py,color='steelblue',edgecolor='white')

# predict classes that are > 0.5, 2,3,5,6,8
# try using only six labels instead of five
submit['labels'] = '1 2 3 5 6 8'
submit.to_csv('naive.csv',index=False)