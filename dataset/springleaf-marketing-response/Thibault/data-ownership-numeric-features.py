import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def rstr(df): return df.apply(lambda x: [x.unique()])

train = pd.read_csv("../input/train.csv", nrows=20000)
nrows = len(train)

# Remove target and ID
for rm_col in ['ID', 'target']:
    if rm_col in train.columns:
        del train[rm_col]

# Drop constant features
card = train.apply(lambda x: x.nunique(dropna=False), axis=0)
cst_features = card[card == 1].index.values
train = train.drop(cst_features, axis=1)

types = train.dtypes
print('# Two types of numeric: ', types[train.dtypes != 'object'].unique())
print('int64: ', len(types[train.dtypes == 'int64']))
print('float64: ', len(types[train.dtypes == 'float64']))

print('')
print('# First pass')
train_int = train.loc[:, train.dtypes == 'int64']
med_max = pd.DataFrame({
    '_min': train_int.apply(lambda x: x.min()),
    '_max': train_int.apply(lambda x: x.max()),
    '_med': train_int.apply(lambda x: x.median())
})
plt.scatter(med_max._med.values, med_max._max.values)
plt.savefig('med_max.png')
plt.clf()
for col in med_max[med_max._med > 100000].index[:5]:
    print(col, train[col].unique())
train.replace(to_replace=[999999997, 999999998, 999999999, 999999996, 999999994], value=np.nan, inplace=True)

print('')
print('# Second pass')
train_int = train.loc[:, train.dtypes == 'int64']
med_max = pd.DataFrame({
    '_min': train_int.apply(lambda x: x.min()),
    '_max': train_int.apply(lambda x: x.max()),
    '_med': train_int.apply(lambda x: x.median())
})
plt.scatter(med_max._med.values, med_max._max.values)
plt.savefig('med_max_2.png')
plt.clf()
for col in med_max[med_max._med > 100000].index[:5]:
    print(col, train[col].unique())
train.replace(to_replace=[999994, 999999], value=np.nan, inplace=True)

print('')
print('# Third pass')
train_int = train.loc[:, train.dtypes == 'int64']
med_max = pd.DataFrame({
    '_min': train_int.apply(lambda x: x.min()),
    '_max': train_int.apply(lambda x: x.max()),
    '_med': train_int.apply(lambda x: x.median())
})
plt.scatter(med_max._med.values, med_max._max.values)
plt.savefig('med_max_3.png')
plt.clf()
for col in med_max[med_max._med < -90000].index[:5]:
    print(col, train[col].unique())
train.replace(to_replace=[-99999], value=np.nan, inplace=True)

print('')
print('# Fourth pass')
train_int = train.loc[:, train.dtypes == 'int64']
med_max = pd.DataFrame({
    '_min': train_int.apply(lambda x: x.min()),
    '_max': train_int.apply(lambda x: x.max()),
    '_med': train_int.apply(lambda x: x.median())
})
plt.scatter(med_max._med.values, med_max._max.values)
plt.savefig('med_max_4.png')
plt.clf()
for col in med_max[med_max._med > 8000].index[:5]:
    print(col, train[col].unique())
train.replace(to_replace=[9999, 9996, 9990, 9998], value=np.nan, inplace=True)

print('')
print('# Fifth pass')
train_int = train.loc[:, train.dtypes == 'int64']
med_max = pd.DataFrame({
    '_min': train_int.apply(lambda x: x.min()),
    '_max': train_int.apply(lambda x: x.max()),
    '_med': train_int.apply(lambda x: x.median())
})
plt.scatter(med_max._med.values, med_max._max.values)
plt.savefig('med_max_5.png')
plt.clf()
for col in med_max[med_max._max > 1000000].index[:5]:
    print(col, train[col].unique())
train.replace(to_replace=[9999999], value=np.nan, inplace=True)

print('')
print('# Sixth pass')
train_int = train.loc[:, train.dtypes == 'int64']
med_max = pd.DataFrame({
    '_min': train_int.apply(lambda x: x.min()),
    '_max': train_int.apply(lambda x: x.max()),
    '_med': train_int.apply(lambda x: x.median())
})
plt.scatter(med_max._med.values, med_max._max.values)
plt.savefig('med_max_6.png')
plt.clf()
for col in med_max[med_max._max > 800000].index[:5]:
    print(col, train[col].unique())
    
print('')
print('# Seventh pass')
train_int = train.loc[:, train.dtypes == 'int64']
med_max = pd.DataFrame({
    '_min': train_int.apply(lambda x: x.min()),
    '_max': train_int.apply(lambda x: x.max()),
    '_med': train_int.apply(lambda x: x.median())
})
plt.scatter(med_max._med.values, med_max._min.values)
plt.savefig('med_max_7.png')
plt.clf()

print('')
print('Study of VAR_1174')
print('Number of 0: ', len(train.VAR_1174[train.VAR_1174 == 0]))
print('Number of non 0: ', len(train.VAR_1174[train.VAR_1174 != 0]))
train.loc[train.VAR_1174 == 0, 'VAR_1174'] = np.nan
percentile = [(i * 1.0) / 100 for i in range(100)]
print(train.VAR_1174.describe())
plt.plot(percentile, train.VAR_1174.quantile(q=percentile))
plt.savefig('VAR_1174_percentile.png')
