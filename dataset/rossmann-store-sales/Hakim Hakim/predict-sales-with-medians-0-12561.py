#!/usr/bin/env python

"Based on dune_dweller script: https://www.kaggle.com/dvasyukova/rossmann-store-sales/predict-sales-with-pandas-py"
"Predict sales as a historical median for a given store, day of week, year and promo"
"This script scores  0.12561 on the public leaderboard"

import pandas as pd

train_file = '../input/train.csv'
test_file = '../input/test.csv'
output_file = 'predictions.csv'

train = pd.read_csv( train_file,parse_dates=[2] )
test = pd.read_csv( test_file,parse_dates=[3] )

# remove rows with zero sales
train = train.loc[train.Sales > 0]

# add year feature 
train['Year'] = [dt.year for dt in train.Date]
test['Year'] = [dt.year for dt in test.Date]
# remove NaNs from Open
test.loc[ test.Open.isnull(), 'Open' ] = 1

columns = ['Store', 'DayOfWeek', 'Year', 'Promo']

medians = train.groupby( columns )['Sales'].median()
medians = medians.reset_index()

test = pd.merge( test, medians, on = columns, how = 'left' )

test.loc[ test.Open == 0, 'Sales' ] = 0

test[[ 'Id', 'Sales' ]].to_csv( output_file, index = False )
