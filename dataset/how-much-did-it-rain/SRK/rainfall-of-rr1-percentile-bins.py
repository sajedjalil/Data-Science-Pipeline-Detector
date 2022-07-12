# -*- coding: utf-8 -*-
"""
Rainfall distribution across various percentile bins of RR1
How much did it rain @ Kaggle
__author__ : SRK
"""
import os
import sys
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

missing_values_list = ["-99900.0", "-99901.0", "-99903.0", "nan", "999.0", "-99000.0"]

def removeMissingValues(val_list):
        """
        Fiunction to remove the missing values from the input list of values. 
        """
        out_list = []
        for val in val_list:
                if val not in missing_values_list:
                        out_list.append(val)
        return out_list

def getRR1Mean(value):
        """
        Function to get variables from RR1 column
        """
        value_list = value.strip().split(' ')
        value_list_wo_na = removeMissingValues(value_list)

        # mean value of non-na values #
        if value_list_wo_na != []:
                value_arr_wo_na = np.array(value_list_wo_na, dtype='float')
                return np.mean(np.abs(value_arr_wo_na))
        else:
                return 0

def plotFig(out_file_name, cumulative=True):
	"""
	Function to plot the rainfall percentage distribution based on RR1 percentile bins
	@param out_file_name : name of the file where the figure needs to be saved
	@param cumulative : whether to compute the cumulative percentage

	With minor modifications, this function could be used for other variables instead of RR1 as well
	"""
	# reading the required columns from train file #
	z = zipfile.ZipFile('../input/train_2013.csv.zip')
	train = pd.read_csv(z.open("train_2013.csv"), usecols=["RR1","Expected"])
	# getting the mean value of RR1 after removing missing values #
	train["RR1"] = train["RR1"].apply( lambda x: getRR1Mean(x) )

	# get the percentile bin values of RR1 #
	RR1 = np.array(train['RR1'][:]).copy()
	exp = np.array(train['Expected'][:]).copy()
	RR1_perc_list = [-0.01,0] + list( np.percentile(RR1[np.where(RR1>0.00000)[0]], list(range(2,101,1)) ))
	dig_RR1 = np.digitize(RR1, RR1_perc_list, right=True)

	# get the probability distribution of rainfall in each of the RR1 percentile bins #
	rain_perc_list = [-1000] + list(range(70)) + [max(train["Expected"])]	
	RR1_prob_arr = np.ones([len(RR1_perc_list)-1, len(rain_perc_list)-1])
	for i,val in enumerate(np.unique(dig_RR1)):
		temp_exp_vals = exp[np.where(dig_RR1==val)[0]]
		temp_dig_exp = np.digitize(temp_exp_vals, rain_perc_list, right=True)
		for j in range(1,len(rain_perc_list)):
			RR1_prob_arr[i][j-1] = ( np.sum(temp_dig_exp == j) / float(len(temp_exp_vals)) )
	if cumulative:
		RR1_prob_arr = np.cumsum(RR1_prob_arr, axis=1)
	RR1_prob_arr = RR1_prob_arr*100


	# create a facet plot using seaborn. code adopted directly from the facet plot code present in seaborn gallery #
	rainfall_mm = np.tile(range(71), 100)
	percentile_bins = np.repeat(range(100), 71)
	df = pd.DataFrame(np.c_[RR1_prob_arr.flat, rainfall_mm, percentile_bins],
                  columns=["Percentage", "Rainfall", "RR1 Bin"])

	grid = sns.FacetGrid(df, col="RR1 Bin", hue="RR1 Bin", col_wrap=10, size=1.5)
	grid.map(plt.axhline, y=0, ls=":", c=".5")
	grid.map(plt.plot, "Rainfall", "Percentage", marker="o", ms=4)
	grid.set(xticks=[0,10,20,30,40,50,60,70], yticks=[0, 25, 50, 75, 100],
         	xlim=(-.5, 72), ylim=(-0.5, 100.5))
	grid.fig.tight_layout(w_pad=1)
	grid.fig.savefig(out_file_name)

if __name__ == "__main__":
	plotFig("Cumulative_rainfall_percentage_for_RR1_percentile_bins1.png", cumulative=True)
	plotFig("Rainfall_percentage_for_RR1_percentile_bins1.png", cumulative=False)