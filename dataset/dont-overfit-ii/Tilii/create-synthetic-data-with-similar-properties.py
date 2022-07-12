__author__ = 'Tilii: https://kaggle.com/tilii7'

# The goal of this script is to make a dataset that is similar to what we have in this competition. By that I mean not only have similar
# target ratios, but also have some features that are useful

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit

rnd = 2019
feats = 300

# Go to https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
# and read about the meaning of all parameters. It is very easy to make the dataset more or less
# challenging by playing with flip_y and class_sep. Setting n_informative and class_sep to higher values
# will also make it less challenging.

X, y = make_classification(n_samples=20000, n_features=feats, n_redundant=15,
      n_informative=10, n_classes=2, weights=( 0.36, 0.64 ), random_state=rnd,
      n_clusters_per_class=1, flip_y=0.075, class_sep=0.5, hypercube=True)

X, y = pd.DataFrame(X), pd.DataFrame(y, columns=['target'])
dataset=pd.concat([y, X], axis=1)
dataset.to_csv('full_simulated.csv', index=False, float_format='%.3f')

sss = StratifiedShuffleSplit(n_splits=1, test_size=250, random_state=101)
for train_index, test_index in sss.split(dataset, dataset.target.values):
    test, train = dataset.loc[train_index].reset_index(drop=True), dataset.loc[test_index].reset_index(drop=True)
train.insert(0, 'id', dataset.index.astype(np.int32)[:250])
train.to_csv('train.csv', index=False, float_format='%.3f')

sss = StratifiedShuffleSplit(n_splits=1, test_size=1975, random_state=999)
for train_index, test_index in sss.split(test, test.target.values):
    test_private, test_public = test.loc[train_index].reset_index(drop=True), test.loc[test_index].reset_index(drop=True)
test = pd.concat([test_public, test_private], axis=0)
test.insert(0, 'id', dataset.index.astype(np.int32)[250:])
test.to_csv('test_targets.csv', index=False, float_format='%.3f')
test.drop(['target'], axis=1, inplace=True)
test.to_csv('test.csv', index=False, float_format='%.3f')

# "Public" test data points are the first 1975 points in the whole test dataset.
# All test targets are in the file test_targets.csv