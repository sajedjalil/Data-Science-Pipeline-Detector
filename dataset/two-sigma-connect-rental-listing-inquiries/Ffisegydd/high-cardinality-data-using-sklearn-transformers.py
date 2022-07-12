# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Load in train and test data
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')

train['train'] = 1
test['train'] = 0

X = pd.concat([train, test])
y = X.pop('interest_level')


class CategoricalAverage(BaseEstimator, TransformerMixin):
    # Example transformer with pieces of code taken from other Python scripts, including:
    # https://www.kaggle.com/jxnlco/two-sigma-connect-rental-listing-inquiries/dealing-with-high-cardinality-data
    # https://www.kaggle.com/stanislavushakov/two-sigma-connect-rental-listing-inquiries/python-version-of-it-is-lit-by-branden
    # https://www.kaggle.com/rakhlin/two-sigma-connect-rental-listing-inquiries/another-python-version-of-it-is-lit-by-branden
    # The original idea originally taken (by all) from it is lit:
    # https://www.kaggle.com/brandenkmurray/two-sigma-connect-rental-listing-inquiries/it-is-lit
    
    def __init__(self, variable, target, fold=5, params=None):
        self.variable = variable
        self.target = target
        self.hcc_name = "_".join(["hcc", self.variable, self.target])
        self.params = params if params is not None else {'f': 1, 'g': 1, 'k': 5}

        self.fold = StratifiedKFold(fold)

    def fit(self, X, y=None):
        self.y = y
        self.dummy = pd.get_dummies(y).astype(int)

        return self

    def transform(self, X, y=None):
        prior = self.dummy.mean()[self.target]

        train_mask = X.train == 1
        X_train = X[train_mask]
        X_test = X[~train_mask]
        self.y = self.y[train_mask]
        self.dummy = self.dummy[train_mask]

        X_train = X_train.join(self.dummy)

        encoding = CategoricalAverage.encode(X_train, prior, self.variable, self.target, self.hcc_name, self.params)
        test_df = X_test[[self.variable]].join(encoding, on=self.variable, how="left")[self.hcc_name].fillna(prior)

        dfs = []
        for train, test in self.fold.split(np.zeros(len(X_train)), self.y):
            train_split = X_train.iloc[train]
            test_split = X_train.iloc[test]
            encoding = CategoricalAverage.encode(train_split, prior, self.variable, self.target, self.hcc_name, self.params)
            df = test_split[[self.variable]].join(encoding, on=self.variable, how="left")[self.hcc_name].fillna(prior)
            dfs.append(df)
        dfs.append(test_df)
        df = pd.concat(dfs)
        df = df.reindex(X.index)

        return df.to_frame(name=self.hcc_name)

    def get_feature_names(self):
        return [self.hcc_name]

    @staticmethod
    def encode(X_train, prior, variable, target, hcc_name, params):
        f, g, k = params['f'], params['g'], params['k']

        grouped = X_train.groupby(variable)[target].agg({"size": "size", "mean": "mean"})
        grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
        grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior
        return grouped

# Example using a single instance
cat_average = CategoricalAverage('building_id', 'high')
transformed = cat_average.fit_transform(X, y)

print(transformed.head())

# Example using 4 separate instances for different variables and targets
feature = FeatureUnion([
    ('building_high', CategoricalAverage('building_id', 'high')),
    ('building_medium', CategoricalAverage('building_id', 'medium')),
    ('manager_high', CategoricalAverage('manager_id', 'high')),
    ('manager_medium', CategoricalAverage('manager_id', 'medium'))
])

transformed = feature.fit_transform(X, y)

print(transformed[:10, :])