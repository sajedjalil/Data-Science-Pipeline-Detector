# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Ensemble of two public kernels
# For the median-based kernel, I add a multiplier of 0.95 to the original result.
# Median-based from Paulo Pinto: https://www.kaggle.com/paulorzp/log-ma-and-days-of-week-means-lb-0-529
# LGBM from Ceshine Lee: https://www.kaggle.com/ceshine/lgbm-starter

# 12/17/2017
# Another submission added to the dataset: https://www.kaggle.com/vrtjso/lgbm-one-step-ahead
# Feel free to try ensembling it! 
# Note that I ran that script locally so the result here may be different from the kernel output.

filepath = '../input/ensemble/'
all_files = os.listdir(filepath)

outs = [pd.read_csv(os.path.join(filepath, f), index_col=0) for f in all_files]
concat_df = pd.concat(outs, axis=1)
concat_df.columns = all_files
concat_df["unit_sales"] = concat_df.mean(axis=1)
concat_df[["unit_sales"]].to_csv("ensemble.csv")