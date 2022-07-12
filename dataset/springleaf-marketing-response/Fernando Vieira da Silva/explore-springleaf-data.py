# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
from pandas import Series

# The competition datafiles are in the directory ../input
# List the files we have available to work with
print("> ls ../input")
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Read train data file:
train = pd.read_csv("../input/train.csv", nrows=10000) # only 10k lines
#train = pd.read_csv("../input/train.csv")

print(train)

variances_values = []
variances_indexes = []
print("---> Ranking variances")
col_names = list(train.columns.values)
for col_name in col_names:
    if col_name == 'target':
        continue
    
    converted_series = train[col_name].convert_objects(convert_numeric='coerce')
    
    
    #num_NaN = converted_series.isnull().sum()
    #if num_NaN > 0:
    #    print("Series %s has %d NaN entries\n" % (col_name, num_NaN))
    
    try:
        variance = converted_series.var()
        print("Variance for %s is %.2f\n" % (col_name, variance))
        variances_values.append(variance)
        variances_indexes.append(col_name)
    except:
        print("Error for obtaining variance for %s" % col_name)
    

var_series = Series(variances_values, index=variances_indexes)

print(var_series)

print(var_series.rank())

print("Mean Variance: %.2f" % var_series.mean())
print("Std deviation variances: %.2f" % var_series.std())