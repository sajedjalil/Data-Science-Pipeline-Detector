#!/usr/bin/env python

"Predict sales as a historical mean for a given store, day of week and promo"

def split_dates(data):
    data['year'] = data.Date.apply(lambda x: x.split('-')[0])
    data['year'] = data['year'].astype(float)
    data['month'] = data.Date.apply(lambda x: x.split('-')[1])
    data['month'] = data['month'].astype(float)
    data['day'] = data.Date.apply(lambda x: x.split('-')[2])
    data['day'] = data['day'].astype(float)

import pandas as pd
import numpy as np

train_file = '../input/train.csv'
test_file = '../input/test.csv'
output_file = 'predictions.csv'

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

# remove NaNs from Open
test.loc[ test.Open.isnull(), 'Open' ] = 1
# remove rows with 0 sales from train
train = train.loc[train.Sales > 0]

print(train.dtypes)
print('Split the dates')
split_dates(train)

#print('Remove November and December')
#train = train[train['month'] != 11]
#train = train[train['month'] != 12]
print('Only use 2015')
train = train[train['year'] == 2015]

print("group by store, day of week and promo and calculate mean sales for each group")
train['Sales'] = np.log1p(train['Sales'])
# convert dates to date datatype
train['Date'] = pd.to_datetime(train['Date'])
print('Get days from first date')
first_date = np.min(train['Date'])
# magic code to change to number of days since first
train['Date'] = (train['Date'] - first_date) / np.timedelta64(1, 'D')
train['Sales'] = train['Sales'] - train['Date'] * 0.00011094
medians = train.groupby([ 'Store', 'DayOfWeek', 'Promo' ])['Sales'].median()
medians = np.expm1(medians)
print(medians.head(6))

print("reset index to get a dataframe with 4 columns")
medians = medians.reset_index()
print(medians.head(6))
print(medians.dtypes)

print("merge with test dataframe to get sales predictions")
test = pd.merge(test, medians, on = ['Store','DayOfWeek','Promo'], how='left')
test.fillna(0, inplace=True)
#test.fillna(train.Sales.mean(), inplace=True)
print(test.head())

test[[ 'Id', 'Sales' ]].to_csv( output_file, index = False )

