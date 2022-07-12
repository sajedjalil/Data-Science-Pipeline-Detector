# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_json('../input/train.json').set_index('listing_id')
test = pd.read_json('../input/test.json').set_index('listing_id')

def add_median_price(key=None, suffix="", trn_df=None, tst_df=None):
    # Set features to be used
    median_features = key.copy()
    median_features.append('price')
    # Concat train and test to find median prices over whole dataset
    median_prices = pd.concat([trn_df[median_features], tst_df[median_features]], axis=0)
    # Group data by key to compute median prices
    medians_by_key = median_prices.groupby(by=key)['price'].median().reset_index()
    # Rename median column with provided suffix
    medians_by_key.rename(columns={'price': 'median_price_' + suffix}, inplace=True)
    # Update data frames, note that merge seems to reset the index
    # that's why I reset first and set again the index
    trn_df = trn_df.reset_index().merge(medians_by_key, on=key, how='left').set_index('listing_id')
    tst_df = tst_df.reset_index().merge(medians_by_key, on=key, how='left').set_index('listing_id')
    trn_df['ratio_' + suffix] = trn_df['price'] /trn_df['median_price_' + suffix]
    tst_df['ratio_' + suffix] = tst_df['price'] / tst_df['median_price_' + suffix]
    trn_df['diff_' + suffix] = trn_df['price'] - trn_df['median_price_' + suffix]
    tst_df['diff_' + suffix] = tst_df['price'] - tst_df['median_price_' + suffix]

    return trn_df, tst_df
print(train.columns)
train, test = add_median_price(key=['bedrooms'],
                               suffix="bed",
                               trn_df=train, tst_df=test)
train_test = pd.concat((train, test), axis=0).reset_index(drop=True)
print(train_test.columns)
train_test.drop([item for item in list(train_test.columns) if item not in ['median_price_bed', 'ratio_bed', 'diff_bed']],axis=1,inplace=True)
train_test.to_csv('bathrooms-price.csv',index=None)