#!/usr/bin/env python

"Predict sales as a historical median for a given store, day of week, and promo"
"This script scores 0.13888 on the public leaderboard"

import pandas as pd

train_file = '../input/train.csv'
test_file = '../input/test.csv'
output_file = 'predictions.csv'

train = pd.read_csv( train_file,low_memory=False )
test = pd.read_csv( test_file ,low_memory=False)

# remove rows with zero sales
# mostly days where closed, but also 54 days when not
train = train.loc[train.Sales > 0]
print (train)
# remove NaNs from Open
test.loc[ test.Open.isnull(), 'Open' ] = 1
test.loc[ test.Promo.isnull(), 'Promo' ] = 1
test.loc[ test.StateHoliday.isnull(), 'StateHoliday' ] = 1
test.loc[ test.SchoolHoliday.isnull(), 'SchoolHoliday' ] = 1

columns = ['Store', 'DayOfWeek', 'Promo','SchoolHoliday','StateHoliday']

medians = train.groupby( columns )['Sales'].median()
medians = medians.reset_index()

test2 = pd.merge( test, medians, on = columns, how = 'left' )
assert( len( test2 ) == len( test ))

test2.loc[ test2.Open == 0, 'Sales' ] = 0
#assert( test2.Sales.isnull().sum() == 0 )

test2[[ 'Id', 'Sales' ]].to_csv( output_file, index = False )

#print( "Up the leaderboard!" )

