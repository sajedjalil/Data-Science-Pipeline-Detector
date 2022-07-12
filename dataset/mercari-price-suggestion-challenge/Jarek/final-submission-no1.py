import pyximport;
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, Imputer

pyximport.install()
import time
import numpy as np
import pandas as pd

from wordbatch.models import FM_FTRL

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import os

os.environ['JOBLIB_TEMP_FOLDER'] = '.'

develop = False


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class FillNa(BaseEstimator, TransformerMixin):
    def __init__(self, value):
        self.value = value

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return X.fillna(self.value)


class LoggingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, message):
        self.message = message

    # noinspection PyUnusedLocal
    def fit(self, X, y=None):
        print("{}: {}, dataset shape: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), self.message,
                                                 X.shape))
        return self

    # noinspection PyMethodMayBeStatic
    def transform(self, X):
        return X


class TopVariance:
    def __init__(self, num_features):
        self.num_features = num_features
        self.vt = VarianceThreshold()

    def fit(self, X, y=None):
        self.vt = self.vt.fit(X)
        if X.shape[1] <= self.num_features:
            threshold = 0.0
        else:
            threshold = np.partition(self.vt.variances_, -self.num_features)[-self.num_features]
        print(threshold)
        self.mask = (self.vt.variances_ > threshold)
        return self

    def transform(self, X):
        return X[:, self.mask]


class LowAlphaNum(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.apply(
            lambda x: ''.join(
                [chr for chr in x.lower() if chr in 'qwertyuiopasdfghjklzxcvbnm1234567890']) if isinstance(x,
                                                                                                           str)
            else '')


class NnzTransformer:
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.mask = np.array(np.clip(X.getnnz(axis=0) - self.threshold, 0, 1), dtype=bool)
        return self

    def transform(self, X):
        return X[:, self.mask]


VARIANCE_THRESHOLD = 1.0e-08

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


pipeline = lambda: Pipeline([
    ('log', LoggingTransformer("Starting transformer/classification pipeline")),
    ('first_union', FeatureUnion([
        ('name', Pipeline([
            ('selector', ItemSelector('name')),
            ('transformer', FillNa('Unknown')),
            ('vectorizer',
             TfidfVectorizer(use_idf=False, min_df=3, token_pattern=r"(?u)\b\w+\b", analyzer='word',
                             ngram_range=(1, 2), strip_accents='ascii')),
            ('log', LoggingTransformer("End of name")),
            ('vt', VarianceThreshold(VARIANCE_THRESHOLD)),
            ('log2', LoggingTransformer("End of name vt"))
        ])),
        ('name2', Pipeline([
            ('selector', ItemSelector('name')),
            ('transformer', FillNa('null')),
            ('lowalphanum', LowAlphaNum()),
            ('vectorizer',
             TfidfVectorizer(use_idf=False, min_df=3, analyzer='char', ngram_range=(2, 6), strip_accents='ascii')),
            ('log', LoggingTransformer("End of name char")),
            ('vt', VarianceThreshold(VARIANCE_THRESHOLD)),
            ('log2', LoggingTransformer("End of name char vt"))
        ])),
        ('item_description', Pipeline([
            ('selector', ItemSelector('item_description')),
            ('transformer', FillNa('No description yet')),
            ('vectorizer',
             TfidfVectorizer(use_idf=False, min_df=3, token_pattern=r"(?u)\b\w+\b", analyzer='word',
                             ngram_range=(1, 2), strip_accents='ascii')),
            ('log', LoggingTransformer("End of item_description")),
            ('vt', VarianceThreshold(VARIANCE_THRESHOLD)),
            ('log2', LoggingTransformer("End of item_description vt"))
        ])),
        ('item_condition_id', Pipeline([
            ('selector', ItemSelector(['item_condition_id'])),
            ('transformer', FillNa(9999)),
            ('ohe', OneHotEncoder(handle_unknown='ignore')),
            ('log', LoggingTransformer("End of item_condition_id"))
        ])),
        ('item_condition_id_2', Pipeline([
            ('selector', ItemSelector(['item_condition_id'])),
            ('imputer', Imputer()),
            ('log', LoggingTransformer("End of item_condition_id_2"))
        ])),
        ('category_name', Pipeline([
            ('selector', ItemSelector('category_name')),
            ('transformer', FillNa('Unknown')),
            ('vectorizer',
             CountVectorizer(token_pattern=r"[^/]+", min_df=3, binary=True, analyzer='word',
                             ngram_range=(1, 5), strip_accents='ascii')),
            ('log', LoggingTransformer("End of category_name"))
        ])),
        ('category_name_2', Pipeline([
            ('selector', ItemSelector('category_name')),
            ('transformer', FillNa('Unknown')),
            ('vectorizer',
             CountVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=3, binary=True, analyzer='word',
                             ngram_range=(1, 2), strip_accents='ascii')),
            ('log', LoggingTransformer("End of category_name 2"))
        ])),
        ('brand_name', Pipeline([
            ('selector', ItemSelector('brand_name')),
            ('transformer', FillNa('Unknown')),
            ('vectorizer', CountVectorizer(token_pattern=r".+", min_df=3, binary=True, analyzer='word',
                                           ngram_range=(1, 1), strip_accents='ascii')),
            ('log', LoggingTransformer("End of brand"))
        ])),
        ('brand_name_2', Pipeline([
            ('selector', ItemSelector('brand_name')),
            ('transformer', FillNa('Unknown')),
            ('vectorizer', CountVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=3, binary=True, analyzer='word',
                                           ngram_range=(1, 2), strip_accents='ascii')),
            ('log', LoggingTransformer("End of brand 2"))
        ])),
        ('shipping', Pipeline([
            ('selector', ItemSelector(['shipping'])),
            ('imputer', Imputer()),
            ('log', LoggingTransformer("End of shipping"))
        ]))
    ], n_jobs=1)),
    ('log2', LoggingTransformer("End of first union")),
    ('log3', LoggingTransformer("Variance eliminated, end of pipeline")),
]
)


def main():
    start_time = time.time()

    print('[{}] Just started :-P'.format(time.time() - start_time))
    train = pd.read_table('../input/train.tsv', engine='c')
    print('Num zero price {}'.format((train.price < 1.0).sum()))
    train = train.drop(train[(train.price < 1.0)].index)
    print('[{}] Finished to load train data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)

    y = np.log1p(train["price"])
    y_mean = np.mean(y)

    pp = pipeline().fit(train, y)
    print('[{}] Pipeline fitted'.format(time.time() - start_time))
    X = pp.transform(train)
    print('[{}] Pipeline transformed'.format(time.time() - start_time))

    train_X, train_y = X, y
    valid_X, valid_y = None, None

    if develop:
        train_X, valid_X, train_y, valid_y = train_test_split(X, y, shuffle=False, test_size=0.05, random_state=100)

    #### FM FTRL ####
    print('[{}] FM FTRL start training'.format(time.time() - start_time))
    fm_ftrl_model = FM_FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=0.1, D=train_X.shape[1], alpha_fm=0.01, L2_fm=0.0,
                            init_fm=0.01,
                            D_fm=200, e_noise=0.0001, iters=18, inv_link="identity", threads=4)
    fm_ftrl_model.fit(train_X, train_y)
    print('[{}] FM FTRL stop training'.format(time.time() - start_time))
    if develop:
        fm_ftrl_preds = fm_ftrl_model.predict(X=valid_X)
        fm_ftrl_preds = np.clip(fm_ftrl_preds, 0, fm_ftrl_preds.max())
        print("[{}] FM FFTRL dev RMSLE:".format(time.time() - start_time),
              rmsle(np.expm1(valid_y), np.expm1(fm_ftrl_preds)))

    # CREATING SUBMISSION
    def predict():
        for test in pd.read_table('../input/test.tsv', engine='c', chunksize=100000):
            try:
                test_X = pp.transform(test)
                preds = fm_ftrl_model.predict(test_X)
                preds = np.clip(preds, 0, preds.max())

                submission = test[['test_id']]
                submission['price'] = np.expm1(preds)
                yield submission
            except:
                submission_mini_chunks = []
                for i in range(test.shape[0]):
                    test_mini_chunk = test.iloc[i:i + 1]
                    try:
                        test_X_mini_chunk = pp.transform(test_mini_chunk)
                        preds_mini_chunk = fm_ftrl_model.predict(test_X_mini_chunk)
                        preds_mini_chunk = np.clip(preds_mini_chunk, 0, preds_mini_chunk.max())
                    except:
                        preds_mini_chunk = [y_mean]

                    submission_mini_chunk = test_mini_chunk[['test_id']]
                    submission_mini_chunk['price'] = np.expm1(preds_mini_chunk)
                    submission_mini_chunks.append(submission_mini_chunk)

                yield pd.concat(submission_mini_chunks)


    print('[{}] Preparing submission'.format(time.time() - start_time))
    final_submission = pd.concat(predict())
    # noinspection PyUnresolvedReferences
    final_submission.to_csv("submission_fm_ftrl_pipeline_chunks.csv", index=False)

    print('[{}] Submission saved'.format(time.time() - start_time))


if __name__ == '__main__':
    main()