# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm, lognorm


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train=pd.read_csv("../input/train.csv")

train['log_loss'] = np.log(train['loss'])

# fit the normal distribution on ln(loss)
(mu, sigma) = norm.fit(train['log_loss'])


#plot loss
n, bins, patches = plt.hist(train['loss'], 60, normed=1, facecolor='green', alpha=0.75)
plt.xlabel('loss')
plt.ylabel('Probability')
plt.grid(True)
plt.savefig('loss.png')

plt.clf()

#plot logloss
n, bins, patches = plt.hist(train['log_loss'], 60, normed=1, facecolor='green', alpha=0.75)
plt.xlabel('log_loss')
plt.ylabel('Probability')
plt.grid(True)

# add the fitted line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)

plt.savefig('log_loss.png')

plt.clf()
