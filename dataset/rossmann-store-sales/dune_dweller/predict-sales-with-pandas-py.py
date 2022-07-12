#!/usr/bin/env python

"Predict sales as a historical mean for a given store, day of week and promo"

import pandas as pd

train_file = '../input/train.csv'
test_file = '../input/test.csv'
output_file = 'predictions.csv'

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

# remove NaNs from Open
test.loc[ test.Open.isnull(), 'Open' ] = 1
# remove rows with 0 sales from train
train = train.loc[train.Sales > 0]

print("group by store, day of week and promo and calculate mean sales for each group")
means = train.groupby([ 'Store', 'DayOfWeek', 'Promo' ])['Sales'].mean()
print(means.head(6))

print("reset index to get a dataframe with 4 columns")
means = means.reset_index()
print(means.head(6))

print("merge with test dataframe to get sales predictions")
test = pd.merge(test, means, on = ['Store','DayOfWeek','Promo'], how='left')
test.fillna(train.Sales.mean(), inplace=True)
print(test.head())

test[[ 'Id', 'Sales' ]].to_csv( output_file, index = False )

