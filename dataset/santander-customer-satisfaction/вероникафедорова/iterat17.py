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

var3838=train.var38.describe()
print(var3838)
train.loc[train['TARGET']==1, 'var38'].describe()
train.var38.hist(bins=1000);
train.var38.map(np.log).hist(bins=1000);
train.var38.value_counts()
var38383=train.var38[train['var38'] != 117310.979016494].mean()
print(var38383)
train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].value_counts()
train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].map(np.log).hist(bins=100);
train['var38mc'] = np.isclose(train.var38, 117310.979016)
train['logvar38'] = train.loc[~train['var38mc'], 'var38'].map(np.log)
train.loc[train['var38mc'], 'logvar38'] = 0
test['var38mc'] = np.isclose(test.var38, 117310.979016)
test['logvar38'] = test.loc[~test['var38mc'], 'var38'].map(np.log)
test.loc[test['var38mc'], 'logvar38'] = 0
var1515=train['var15'].describe()
print(var1515)
train['var15'].hist(bins=100);
train['var15'].value_counts()

sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(sns.kdeplot, "var15") \
   .add_legend()
plt.title('Unhappy customers are slightly older');
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(plt.hist, "var15") \
   .add_legend()
   
train['log_saldo_var30'] = train.saldo_var30.map(np.log)
test['log_saldo_var30'] = test.saldo_var30.map(np.log)
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(sns.kdeplot, "saldo_var30") \
   .add_legend();
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(sns.kdeplot, "log_saldo_var30") \
   .add_legend();
sns.FacetGrid(train, hue="TARGET", size=10) \
   .map(plt.scatter, "var38", "var15") \
   .add_legend();
sns.FacetGrid(train, hue="TARGET", size=10) \
   .map(plt.scatter, "logvar38", "var15") \
   .add_legend()
plt.ylim([0,120]);
sns.FacetGrid(train[~train.var38mc], hue="TARGET", size=10) \
   .map(plt.scatter, "logvar38", "var15") \
   .add_legend()
plt.ylim([0,120]);
sns.FacetGrid(train[train.var38mc], hue="TARGET", size=6) \
   .map(sns.kdeplot, "var15") \
   .add_legend();
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(sns.kdeplot, "SumZeros") \
   .add_legend()
plt.title('Unhappy customers have a lot of features that are zero');
var3636=train['var36'].value_counts()
print(var3636)
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(sns.kdeplot, "var36") \
   .add_legend()
plt.title('If var36 is 0,1,2 or 3 => less unhappy customers');
sns.FacetGrid(train[(~train.var38mc) & (train.var36 < 4)], hue="TARGET", size=10) \
   .map(plt.scatter, "var36", "logvar38") \
   .add_legend();
sns.FacetGrid(train[(~train.var38mc) & (train.var36 ==99)], hue="TARGET", size=6) \
   .map(sns.kdeplot, "logvar38") \
   .add_legend();
var55=train.num_var5.value_counts() 
print(var55)
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(plt.hist, "num_var5") \
   .add_legend();
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(sns.kdeplot, "num_var5") \
   .add_legend();
sns.pairplot(train[['var15', 'var36', 'logvar38', 'TARGET']], hue="TARGET", size=2, diag_kind="kde");
sns.FacetGrid(train[train['var3'] != -999999], hue="TARGET", size=10) \
   .map(sns.kdeplot, "var3") \
   .add_legend();
   

import pandas as pd
import numpy as np


def intersect(a, b):
    return list(set(a) & set(b))


def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    output.remove('ID')
    return output


def prepare_dataset(train, test):   
    features = train.columns.values
    
    norm_f = []
    for f in features:
        u = train[f].unique()
        if len(u) != 1:
            norm_f.append(f)


    remove = []
    for i in range(len(norm_f)):
        v1 = train[norm_f[i]].values
        for j in range(i+1, len(norm_f)):
            v2 = train[norm_f[j]].values
            if np.array_equal(v1, v2):
                remove.append(norm_f[j])
    
    for r in remove:
        norm_f.remove(r)

    train = train[norm_f]
    norm_f.remove('TARGET')
    test = test[norm_f]
    features = get_features(train, test)
    return train, test, features


def find_min_max_features(df, f):
    return df[f].min(), df[f].max()


def analayze_data(train, test):
    print('Length of train: ', len(train.index))
    train_zero = train[train['TARGET'] == 0]
    print('Length of train [TARGET = 0]: ', len(train_zero.index))
    train_one = train[train['TARGET'] == 1]
    print('Length of train [TARGET = 1]: ', len(train_one.index))
    # train_one.to_csv("debug.csv", index=False)
    one_range = dict()
    for f in train.columns:
        mn0, mx0 = find_min_max_features(train_zero, f)
        mn1, mx1 = find_min_max_features(train_one, f)
        mnt = 'N/A'
        mxt = 'N/A'
        if f in test.columns:
            mnt, mxt = find_min_max_features(test, f)
        one_range[f] = (mn1, mx1)
        if mn0 != mn1 or mn1 != mnt or mx0 != mx1 or mx1 != mxt:
            print("\nFeature {}".format(f))
            print("Range target=0  ({} - {})".format(mn0, mx0))
            print("Range target=1  ({} - {})".format(mn1, mx1))
            print("Range in test   ({} - {})".format(mnt, mxt))


train, test, features = prepare_dataset(train, test)
analayze_data(train, test)