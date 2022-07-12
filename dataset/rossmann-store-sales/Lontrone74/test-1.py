#!/usr/bin/env python

"Predict sales as a historical mean for a given store, day of week, an so on"

import pandas as pd

train_file = '../input/train.csv'
test_file = '../input/test.csv'
output_file = 'predictions.csv'

#

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

# remove NaNs from test.Open
test.loc[ test.Open.isnull(), 'Open' ] = 1

store_day_open_promo_holiday = [ 'Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday' ]
store_day_open = [ 'Store', 'DayOfWeek', 'Open' ]	# backup for rows not covered by the first set

# this fast version with pre-computed medians and merge thanks to dune_dweller
# https://www.kaggle.com/dvasyukova/rossmann-store-sales/predict-sales-with-pandas-py/code

medians = train.groupby( store_day_open_promo_holiday )['Sales'].median()
medians = medians.reset_index()

medians_backup = train.groupby( store_day_open )['Sales'].median()
medians_backup = medians_backup.reset_index()

test2 = pd.merge( test, medians, on = store_day_open_promo_holiday, how = 'left' )
test2_backup = pd.merge( test, medians_backup, on = store_day_open, how = 'left' )

assert( len( test2 ) == len( test2_backup ))

# apply backup
test2.loc[ test2.Sales.isnull(), 'Sales' ] = test2_backup.loc[ test2.Sales.isnull(), 'Sales' ]

# shop closed -> sales = 0
test2.loc[ test2.Open == 0, 'Sales' ] = 0

assert( test2.Sales.isnull().sum() == 0 )
test2[[ 'Id', 'Sales' ]].to_csv( output_file, index = False )

print( "Up the leaderboard!" )