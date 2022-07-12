import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

def rstr(df): return df.apply(lambda x: [x.unique()])

# Read train data file:
train = pd.read_csv("../input/train.csv", nrows=20000)
nrows = len(train)

# Remove target and ID
for rm_col in ['ID', 'target']:
    if rm_col in train.columns:
        del train[rm_col]

def is_real(obj):
    try:
        if np.isnan(obj):
            return False
    except TypeError:
        return False
    if np.isreal(obj):
        return True
    try:
        float(obj)
        return True
    except ValueError:
        return False

# _is_str = train.applymap(lambda x: type(x) == str)
# _is_real = train.applymap(is_real)
# n_str = _is_str.sum(0)
# n_real = _is_real.sum(0)
# Columns with mix types
# n_str = n_str[(n_str > 0) & (n_real > 0)]
# print('mix type column:', len(n_str))

# Lot of columns, let's figure out which are categorial
# We don't want to drop NA because column with NA, other has to be kept
card = train.apply(lambda x: x.nunique(dropna=False), axis=0)
# h = plt.hist(card / nrows, bins=100)
# plt.savefig('features_cardinality.png')
cst_features = card[card == 1].index.values
print('## Constant feature count: ', len(cst_features))
print('## ', cst_features)
train = train.drop(cst_features, axis=1)

# Let's look at non numeric column
print('')
print('## Let us look at non numeric column:')
print('## 1. -1 <=> NaN')
print('## 2. Boolean / Categorial columns')
print('## 3. Date type columns')
types = train.dtypes
str_features = types[train.dtypes == 'object'].index.values
print(rstr(train[str_features]))

# Remove -1
for f in str_features:
    train.loc[train[f] == '-1', f] = np.nan

# Detect date column : try to parse first non null value
date_fmt = "%d%b%y:%H:%M:%S"
date_features = []
for f in str_features:
    val = train.loc[train[f].notnull(), f].values[0]
    try:
        time.strptime(val, date_fmt)
        date_features.append(f)
    except (ValueError, TypeError):
        pass

print('')
print('## Date features: ', len(date_features))
print('## ', date_features)
for f in date_features:
    train[f] = pd.to_datetime(train[f], format=date_fmt)

print('')
print('## Cardinality: ')
str_features = list(set(str_features).difference(set(date_features)))
str_card = train[str_features].apply(lambda x: x.nunique(dropna=False))
plt.hist(str_card)
plt.savefig('features_cardinality.png')
plt.clf()
str_card[str_card > .1 * nrows]
print('# > 1000: ', str_card[str_card > 1000])
print('# VAR_0200 is an address.')
str_card = str_card.drop(['VAR_0200'])
plt.hist(str_card)
plt.savefig('features_cardinality_1.png')
plt.clf()
print('# > 100: ', str_card[str_card > 100].index.values)
print('# VAR_0493: ', train['VAR_0493'].dropna().values)
print('# VAR_0404: ', train['VAR_0404'].dropna().values)
print('# VAR_0493 & VAR_0404 seem to be job titles.')
str_card = str_card.drop(['VAR_0493', 'VAR_0404'])
plt.hist(str_card)
plt.savefig('features_cardinality_2.png')
plt.clf()
print('# > 20: ', str_card[str_card > 20].index.values)
print('# VAR_0237 & VAR_0274 are US states abbreviations.')
str_card = str_card.drop(['VAR_0237', 'VAR_0342', 'VAR_0274'])
plt.hist(str_card)
plt.savefig('features_cardinality_3.png')
plt.clf()
print('# Features with only 2 modes:')
for col in str_card[str_card <= 2].index.values:
    print(col, train[col].unique(), '#Not NaN : ',len(train[col].dropna()))
print('# Features with more than 2 modes:')
for col in str_card[str_card > 2].index.values:
    print(col, train[col].unique())
    
print('')
print('## Wrap-up: ')
print('1. VAR_0200 = addresses: to be processes individually ')
print('2. VAR_0493 & VAR_0404 = job titles: may be to be cleaned a little bit ')
print('3. VAR_0237 & VAR_0274 = US States: check for invalid states ')
print('4. VAR_0342 = ?: left as-is ')
print('5. To be removed: left as-is ')
print('   VAR_0239 almost constant')
print('   VAR_0196 almost constant')
print('   VAR_0009 almost constant')
print('   VAR_0229 almost constant')
print('   VAR_0010 almost constant')
print('   VAR_0043 almost constant')
print('   VAR_0012 almost constant')
print('   VAR_0008 almost constant')
print('   VAR_0011 almost constant')
print('   VAR_0202 almost constant')
print('   VAR_0044 almost constant')
print('   VAR_0222 almost constant')
print('   VAR_0216 almost constant')
print('   VAR_0214 almost constant')
print('VAR_0466 : Create indicatrice')
print('6. Boolean features:')
print('   VAR_0232')
print('   VAR_0226')
print('   VAR_0236')
print('   VAR_0230')


# Find Boolean
# Find categorical
# Find date column
# ...
# Redress string
# remove integer -99999 (histo)

