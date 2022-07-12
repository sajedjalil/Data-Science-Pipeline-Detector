"""
Caterpillar @ Kaggle
Adapted from arnaud demytt's R script
AND
Gilberto Titericz Junior's python scripts
__author__ = saihttam
"""


import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import ShuffleSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence

np.random.seed(42)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# load training datasets
train = pd.read_csv(os.path.join('..', 'input', 'train_set.csv'), parse_dates=[2,])
tube_data = pd.read_csv(os.path.join('..', 'input', 'tube.csv'))

train = pd.merge(train, tube_data, on='tube_assembly_id')

# create some new features
train['year'] = train.quote_date.dt.year
train['month'] = train.quote_date.dt.month
train['week'] = train.quote_date.dt.dayofyear % 52

train = train.drop(['quote_date', 'tube_assembly_id'], axis=1)
rs = ShuffleSplit(train.shape[0], n_iter=3, train_size=.2, test_size=.8, random_state=0)
for train_index, _ in rs:
    pass

train = train.iloc[train_index]
print(train.shape)
# Adapted from R script, only use top {threshold features from categorical columns}
newdf = train.select_dtypes(include=numerics)
numcolumns = newdf.columns.values

allcolumns = train.columns.values
nonnumcolumns = list(set(allcolumns) - set(numcolumns))
print("Numcolumns %s " % numcolumns)
print("Nonnumcolumns %s " % nonnumcolumns)

print("Nans before processing: \n {0}".format(train.isnull().sum()))
train[numcolumns] = train[numcolumns].fillna(-999999)
train[nonnumcolumns] = train[nonnumcolumns].fillna("NAvalue")
print("Nans after processing: \n {0}".format(train.isnull().sum()))

for col in nonnumcolumns:
    ser = train[col]
    counts = ser.value_counts().keys()
    # print "%s has %d different values before" % (col, len(counts))
    threshold = 5
    if len(counts) > threshold:
        ser[~ser.isin(counts[:threshold])] = "rareValue"
    if len(counts) <= 1:
        print("Dropping Column %s with %d values" % (col, len(counts)))
        train = train.drop(col, axis=1)
    else:
        train[col] = ser.astype('category')

train = pd.get_dummies(train)
print("Size after dummies {0}".format(train.shape))

# Use log for some variables for better visualization
train["logquantity"] = np.log(train['quantity'])
train["log1usage"] = np.log1p(train['annual_usage'])
train["log1radius"] = np.log1p(train['bend_radius'])
train["log1length"] = np.log1p(train['length'])
train = train.drop(['quantity', 'annual_usage', 'bend_radius', 'length'], axis=1)

labels = train.cost.values
Xtrain = train.drop(['cost'], axis=1)
names = list(Xtrain.columns.values)
Xtrain = np.array(Xtrain)

label_log = np.log1p(labels)
Xtrain, label_log = shuffle(Xtrain, label_log, random_state=666)

model = ExtraTreesClassifier(n_estimators=50, max_depth=15)
model.fit(Xtrain, label_log)
features = []

# display the relative importance of each attribute
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(len(importances)):
    print("%d. feature %d (%f), %s" % (f + 1, indices[f], importances[indices[f]], names[indices[f]]))
    features.append(indices[f])
    # Print only first 5 most important variables
    if len(features) >= 5:
        break

q = pd.qcut(train["cost"], 5)
print("Bins are {0}".format(q))
train['cost_5'] = q

fig = plt.figure()
featurenames = [names[feature] for feature in features]
featurenames.append('cost_5')
pg = sns.pairplot(train[featurenames], hue='cost_5', size=2.5)
pg.savefig('pairplotquintile.png')


print("Training GBRT...")
clf = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                learning_rate=0.1, loss='huber',
                                random_state=1)
clf.fit(Xtrain, label_log)
print('Convenience plot with ``partial_dependence_plots``')

# 2-D dependence plot
target_feature = (features[0], features[1])
features.append(target_feature)
fig, axs = plot_partial_dependence(clf, Xtrain, features, feature_names=names,
                                   n_jobs=3, grid_resolution=50)
fig.savefig('partial.png')