# -*- coding: utf-8 -*-
"""
Scatter plot of rainfall given in the training set
__author__ : SRK
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

# reading the file as dataframe and getting summary #
train_file = "../input/train.csv"
train_df = pd.read_csv(train_file, usecols = ['Id','Expected'])
print(train_df['Expected'].describe())

# grouping the rows based on id and get the rainfall #
train_df_grouped = train_df.groupby(['Id'])
exp_rainfall = np.sort(np.array(train_df_grouped['Expected'].aggregate('mean')))

# plotting a scatter plot #
plt.figure()
plt.scatter(np.arange(exp_rainfall.shape[0]), exp_rainfall)
plt.title("Scatterplot for Rainfall distribution in train sample")
plt.ylabel("Rainfall in mm")
plt.savefig("ExpectedRainfall.png")
plt.show()