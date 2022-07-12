# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Plotting Utilities
from sklearn.linear_model import LogisticRegression
from scipy import interpolate
from sklearn import tree

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')
train_data = train_data.dropna()
targets = train_data['target']
indices = ['v' + str(i) for i in range(1,132,1)]
print(indices)
x = train_data[indices]
print(targets)
print(x)
# Exploring the data
# Let's begin by plotting the first variable and seeing if there are any apparent discrepancies
# between the 0 and 1 data. We will do this by looking at the distribution of the data using a 
# histogram for the 0's and 1's

# Setting a few parameters
num_bins = 50

for jj,ii in enumerate(indices):
    if jj < 10:
        if x[ii].dtype != object:
            f = plt.figure()
            n_0,bins_0,patches_0 = plt.hist(x[ii][targets == 0],num_bins,normed = 1,\
                                        facecolor = 'green',alpha = 0.5)
            n_1,bins_1,patches_1 = plt.hist(x[ii][targets == 1],num_bins,normed = 1,\
                                        facecolor = 'blue',alpha = 0.5)
            plt.title('feature: ' + str(ii))
            

