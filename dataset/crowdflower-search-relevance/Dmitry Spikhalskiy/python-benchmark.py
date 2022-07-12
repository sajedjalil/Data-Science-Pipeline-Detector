from numpy.core.multiarray import ndarray
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors.dist_metrics import DistanceMetric
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup


class LsaMapper:

    tfv = TfidfVectorizer(min_df=3, max_features=None,
                      strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                      ngram_range=(1, 3), smooth_idf=1, sublinear_tf=1,
                      stop_words='english')
    svd = TruncatedSVD(n_components=100, n_iter=15)
    scl = StandardScaler()

    def fit(self, dataset, y=None):
        prod_data = list(dataset.apply(
            lambda x: '%s %s' % (x['product_title'], x['product_description']),
            axis=1))
        X = self.tfv.fit_transform(prod_data)
        X = self.svd.fit_transform(X)
        self.scl.fit(X)
        return self

    def transform(self, dataset):
        assert isinstance(dataset, pd.DataFrame)
        prod_data = list(dataset.apply(
            lambda x: '%s %s' % (x['product_title'], x['product_description']),
            axis=1))
        query_data = list(dataset.apply(lambda x: str(x['query']), axis=1))

        X = self.tfv.transform(prod_data)
        Q = self.tfv.transform(query_data)

        X = self.svd.transform(X)
        Q = self.svd.transform(Q)

        X = self.scl.fit_transform(X)
        Q = self.scl.transform(Q)

        return prepare_features(X, Q)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def prepare_features(X, Q):
    assert isinstance(X, ndarray)
    assert isinstance(Q, ndarray)
    dist = DistanceMetric.get_metric('euclidean')
    assert isinstance(dist, DistanceMetric)
    pairwise_distance = dist.pairwise(X, Q)
    assert isinstance(pairwise_distance, ndarray)
    diagonal = pairwise_distance.diagonal()
    assert isinstance(diagonal, ndarray)
    diagonal = np.reshape(diagonal, (-1, 1))
    return np.concatenate([  # X, Q,
                             X - Q, diagonal], axis=1)

def drop_html(html):
    return BeautifulSoup(html).get_text(separator=" ")

def clean_html(line):
    line[1] = drop_html(str(line[1]))
    line[2] = drop_html(str(line[2]))


def clean_html(line):
    line[1] = drop_html(str(line[1]))
    line[2] = drop_html(str(line[2]))


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# we dont need ID columns
idx = test.id.values.astype(int)
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

# remove html
train.apply(clean_html, axis=1, raw=True)  # decreased results
test.apply(clean_html, axis=1, raw=True)

# create labels. drop useless columns
y = train.median_relevance.values

train = train.drop(['median_relevance', 'relevance_variance'], axis=1)


featureMapper = LsaMapper()
featureMapper.fit(pd.concat([train, test]))

classifier = RandomForestClassifier(n_estimators=25)

classifier.fit(featureMapper.transform(train), y)
predictions = classifier.predict(featureMapper.transform(test))

submission = pd.DataFrame({"id": idx, "prediction": predictions})
submission.to_csv("beating_the_benchmark_yet_again.csv", index=False)
    

