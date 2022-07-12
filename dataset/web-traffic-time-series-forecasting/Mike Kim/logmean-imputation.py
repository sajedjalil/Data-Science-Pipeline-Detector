# Michael S Kim (Python3)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(0)


# Read in data
# can't write out all rows
# The kernel used almost all available (536870912) disk space. Disk writes might have failed due to insufficient space.
data = pd.read_csv("../input/train_1.csv", nrows=2000)
print(data.head(5))

date_cols = data.columns.values.tolist()
date_cols.pop(0)
date_cols[0:5]


# log mean function
def logmean(x):
    return np.expm1(np.mean(np.log1p(x)))


# calculate global log mean to impute fully empty series
logmean_df = data.drop('Page', 1)
logmean_df.fillna(value=0, inplace=True)
logmean_df = logmean_df.values.flatten()
LOGMEAN = logmean(logmean_df)
print(LOGMEAN)


# fills each row
def mean_fill(x):
    if np.sum(np.isnan(x)) == len(x):
        return LOGMEAN
    else:
        x = x[~np.isnan(x)]
        return logmean(x)


# tests
mean_fill(np.array([1,2,3]))
mean_fill(np.array([np.nan,np.nan,np.nan]))



# drop page before row based logmean imputation
logmean_df = data.drop('Page', 1)
impute_values = logmean_df.apply(mean_fill, axis=1)
data['imputed_values'] = impute_values
print(data.head(5))


# fill each row with different value
data[date_cols] = data[date_cols].apply(lambda x: x.fillna(value=data['imputed_values']))
print(data.head(5))

# write back to csv
data.drop('imputed_values', 1, inplace=True)
print(data.head(5))
data.to_csv("train_1_imputed.csv")

