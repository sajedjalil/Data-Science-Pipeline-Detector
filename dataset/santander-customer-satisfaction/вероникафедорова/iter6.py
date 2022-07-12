import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape)
print(test.shape)
print(train.drop_duplicates(subset=train.columns[1:-1]).shape)
print(train.drop_duplicates(subset=train.columns[1:]).shape)
print(train.drop_duplicates(subset=train.columns[1:-1], keep=False).shape)
print(train.drop_duplicates(subset=train.columns[1:], keep=False).shape)
duplicate_ids = set(train['ID']).difference(set(train.drop_duplicates(subset=train.columns[1:-1], keep=False)['ID']))
duplicate_ids_2 = set(train['ID']).difference(set(train.drop_duplicates(subset=train.columns[1:], keep=False)['ID']))
print(len(duplicate_ids))
print(len(duplicate_ids_2))
to_drop = duplicate_ids.difference(duplicate_ids_2)
len(to_drop)
train = train[~train['ID'].isin(to_drop)].drop_duplicates(subset=train.columns[1:])
train.shape
print(train.shape)
df = pd.DataFrame(train.TARGET.value_counts())
df['Percentage'] = 100*df['TARGET']/train.shape[0]
print(df)
train.var3.value_counts()[:10]
features = train.columns[1:-1]
train.insert(1, 'SumZeros', (train[features] == 0).astype(int).sum(axis=1))
test.insert(1, 'SumZeros', (test[features] == 0).astype(int).sum(axis=1))
train.num_var4.hist(bins=100)
plt.xlabel('Number of bank products')
plt.ylabel('Number of customers in train')
plt.title('Most customers have 1 product with the bank')
plt.show()
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(plt.hist, "num_var4") \
   .add_legend()
plt.title('Unhappy cosutomers have less products')
plt.show()
train[train.TARGET==1].num_var4.hist(bins=6)
plt.title('Amount of unhappy customers in function of the number of products');
train.var38.describe()