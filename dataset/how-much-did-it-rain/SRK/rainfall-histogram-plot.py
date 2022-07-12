# -*- coding: utf-8 -*-
"""
Rainfall Histogram Plot
How much did it rain @ Kaggle
__author__ : SRK
"""

# Importing the necessary modules #
import numpy as np
import pandas as pd
from pandas.io.common import ZipFile
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the expected rainfall column from train file #
with ZipFile("../input/train_2013.csv.zip") as z:
     f = z.open('train_2013.csv')
train_exp = pd.read_csv(f,usecols=['Expected'])['Expected']

# Create bins on rainfall for 0mm, 1mm, 2mm,....69mm, >=70mm #
rain_bins_list = [-1] + list( range(70) ) + [max(train_exp)+1]
dig_RR1 = np.digitize(train_exp, rain_bins_list, right=True)
dig_RR1 = dig_RR1-1     # subtracting by 1 to have the pythonic index for digitized values

# plot a histogram on the log count of occurrences #
hist_bins = [-1]+list( range(max(dig_RR1)+2) )
n, bins, patches = plt.hist(dig_RR1, hist_bins, histtype='bar', rwidth=0.8, log=True, align='left', color='green')

# Change the x ticks and use a custom name #
name = [str(i) for i in range(max(dig_RR1))]+['>=70']
xticks = plt.xticks(range(max(dig_RR1)+1), name, size='small')

# Set the title and labels for plot #
plt.title("Histogram of Rainfall in the train set", fontsize="large", fontweight="bold")
plt.xlabel("Rainfall in mm", fontsize="medium", fontweight="bold")
plt.ylabel("Log of Number of Occurrences", fontsize="medium", fontweight="bold")
plt.savefig("output.png")
#plt.show()
plt.close()
