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
means = train.groupby([ 'Store', 'DayOfWeek' ])['Sales'].mean()
print(means.head(6))
