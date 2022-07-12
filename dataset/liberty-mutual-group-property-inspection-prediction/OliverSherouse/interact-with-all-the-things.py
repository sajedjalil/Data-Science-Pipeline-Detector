import logging

import numpy as np
import pandas as pd
import sklearn.cross_validation
import sklearn.decomposition
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.base

from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)

class CategoricalIntegerTransformer(sklearn.base.TransformerMixin):
    def __init__(self, categoricals):
        super().__init__()
        self.categoricals = categoricals
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        return np.vstack([i if isinstance(i[0], int)
                          else np.array([ord(j) for j in i])
                          for i in X.T
                         ]).T
    
class DenseTransformer(sklearn.base.TransformerMixin):     
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        return X.toarray()


class InteractionTransformer(sklearn.base.TransformerMixin):
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        return np.hstack([
                (X[:,i] * X[:,i:].T).T
                for i in range(X.shape[1])
            ])
            

def gini_score(y, y_pred):
    assert(y.shape == y_pred.shape)
    n = y.shape[0]
    sorted_y = y[np.argsort(y_pred)[::-1]]
    total = y.sum()
    chunks = sorted_y / total / n
    nulls = np.array([1 / n / n for i in range(n)])
    return chunks.cumsum().sum() - nulls.cumsum().sum()


def normalized_gini_score(y, y_pred):
    return gini_score(y, y_pred) / gini_score(y, y)
    


data = pd.read_csv("../input/train.csv", index_col=0)
X, y = data.values[:,1:], data.values[:,0]
categoricals = np.array([isinstance(i, str) for i in X[0]])


pl = sklearn.pipeline.Pipeline(steps=(
        ("intcat", CategoricalIntegerTransformer(categoricals)),
        ("onehot", sklearn.preprocessing.OneHotEncoder(categorical_features=categoricals)),
        ("dense", DenseTransformer()),
        ("interaction", InteractionTransformer()),
        #("reduce", sklearn.decomposition.PCA(n_components=10)),

        #("model", sklearn.linear_model.Ridge()),
        ("model", sklearn.ensemble.RandomForestRegressor(n_estimators=100)),

))

scores = sklearn.cross_validation.cross_val_score(
    pl, X, y, cv=5, verbose=10,
    scoring=sklearn.metrics.make_scorer(normalized_gini_score))
    
logging.info("{:.1%} ({:.1%})".format(scores.mean(), scores.std()))