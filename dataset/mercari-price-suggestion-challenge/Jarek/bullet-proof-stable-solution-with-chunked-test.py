import pyximport;
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder

pyximport.install()
import time
import numpy as np
import pandas as pd
import os

from wordbatch.models import FM_FTRL

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# allowing FeatureUnion to use the scratch surface
os.environ['JOBLIB_TEMP_FOLDER'] = '.'
dev_env = False


def log(message):
    print("{}: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), message))


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

    def fit(self, X, y=None):
        print("{}: {}, dataset shape: {}".format(time.strftime("%Y-%m-%d %H:%M:%S",
                                                               time.gmtime()), self.message, X.shape))
        return self

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
        log("TopVariance established threshold: {}".format(threshold))
        self.mask = (self.vt.variances_ > threshold)
        return self

    def transform(self, X):
        return X[:, self.mask]


class MixedTfidfVectorizer:
    # this class is a char based vectorizer that filters char-sequences only to existing n-grams
    # should fix the ambiguities like iphone 6 256 GB vs. iphone6 256GB
    # hacky implementation works only with lower-case characters hard-coded in one of funcs.
    def __init__(self, char_ngram_range=(2, 10), **kwargs):
        # popping the one new field
        self.char_ngram_range = char_ngram_range
        self.kwargs = kwargs
        self.tfidf_chars = None

    @staticmethod
    def get_alphanum_seq(x):
        return ''.join(
            [chr for chr in x.lower() if chr in
             'qwertyuiopasdfghjklzxcvbnm1234567890+']) \
            if isinstance(x, str) else ''

    def get_alphanum_seqs(self, X):
        return X.apply(self.get_alphanum_seq)

    def fit(self, X, y=None):
        tfidf_words = TfidfVectorizer(**self.kwargs)
        tfidfv = tfidf_words.fit(X)
        X_alphanum = self.get_alphanum_seqs(X)
        # a hacky way to convert vocabulary
        vocabulary = frozenset(
            {self.get_alphanum_seq(''.join(v.split()))[:self.char_ngram_range[1]] \
             for v in tfidfv.vocabulary_})
        kwargs = dict(self.kwargs)
        kwargs['ngram_range'] = self.char_ngram_range
        kwargs['vocabulary'] = vocabulary
        kwargs['analyzer'] = 'char'
        self.tfidf_chars = TfidfVectorizer(**kwargs).fit(X_alphanum)
        return self

    def transform(self, X):
        return self.tfidf_chars.transform(self.get_alphanum_seqs(X))


VARIANCE_THRESHOLD = 1.42785096141e-08


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


pipeline = lambda: Pipeline([
    ('log', LoggingTransformer("Starting feature extraction pipeline")),
    ('first_union', FeatureUnion([
        ('name', Pipeline([
            ('selector', ItemSelector('name')),
            ('transformer', FillNa('Unknown')),
            ('vectorizer',
             TfidfVectorizer(use_idf=False, min_df=10, token_pattern=r"(?u)\b\w+\b",
                             analyzer='word', ngram_range=(1, 2), strip_accents='ascii')),
            ('log', LoggingTransformer("End of name")),
            ('vt', VarianceThreshold(VARIANCE_THRESHOLD)),
            ('log2', LoggingTransformer("End of name vt"))
        ])),
        ('name2', Pipeline([
            ('selector', ItemSelector('name')),
            ('transformer', FillNa('Unknown')),
            ('vectorizer',
             MixedTfidfVectorizer(use_idf=False, min_df=10, token_pattern=r"(?u)\b\w+\b",
                                  analyzer='word', ngram_range=(1, 2), strip_accents='ascii')),
            ('log', LoggingTransformer("End of name2")),
            ('vt', VarianceThreshold(VARIANCE_THRESHOLD)),
            ('log2', LoggingTransformer("End of name2 vt"))
        ])),
        ('item_description', Pipeline([
            ('selector', ItemSelector('item_description')),
            ('transformer', FillNa('No description yet')),
            ('vectorizer',
             TfidfVectorizer(use_idf=False, min_df=10, token_pattern=r"(?u)\b\w+\b",
                             analyzer='word', ngram_range=(1, 2), strip_accents='ascii')),
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
            ('log', LoggingTransformer("End of item_condition_id_2"))
        ])),
        ('category_name', Pipeline([
            ('selector', ItemSelector('category_name')),
            ('transformer', FillNa('Unknown')),
            ('vectorizer',
             CountVectorizer(token_pattern=r"[^/]+", min_df=3, binary=True,
                             analyzer='word', ngram_range=(1, 5), strip_accents='ascii')),
            ('log', LoggingTransformer("End of category_name"))
        ])),
        ('category_name_2', Pipeline([
            ('selector', ItemSelector('category_name')),
            ('transformer', FillNa('Unknown')),
            ('vectorizer',
             CountVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=3, binary=True,
                             analyzer='word', ngram_range=(1, 2), strip_accents='ascii')),
            ('log', LoggingTransformer("End of category_name 2"))
        ])),
        ('brand_name', Pipeline([
            ('selector', ItemSelector('brand_name')),
            ('transformer', FillNa('Unknown')),
            ('vectorizer', CountVectorizer(token_pattern=r".+", min_df=3, binary=True,
                                           analyzer='word', ngram_range=(1, 1),
                                           strip_accents='ascii')),
            ('log', LoggingTransformer("End of brand"))
        ])),
        ('brand_name_2', Pipeline([
            ('selector', ItemSelector('brand_name')),
            ('transformer', FillNa('Unknown')),
            ('vectorizer', CountVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=3, binary=True,
                                           analyzer='word', ngram_range=(1, 2),
                                           strip_accents='ascii')),
            ('log', LoggingTransformer("End of brand 2"))
        ])),
        ('shipping', Pipeline([
            ('selector', ItemSelector(['shipping'])),
            ('log', LoggingTransformer("End of shipping"))
        ]))
    ], n_jobs=-1)),
    ('log2', LoggingTransformer("End of first union")),
    ('vt', VarianceThreshold(VARIANCE_THRESHOLD)),
    ('var', TopVariance(3000000)),
    ('log3', LoggingTransformer("Variance eliminated, end of pipeline")),
]
)


def main():
    start_time = time.time()

    # DATA IN
    log('Start')
    train = pd.read_table('../input/train.tsv', engine='c')
    log('Dropping elements with 0 price, number affected: {}'.format((train.price < 1.0).sum()))
    train = train.drop(train[(train.price < 1.0)].index)

    log('Train data loaded'.format(time.time() - start_time))
    log('Train shape: {}'.format(train.shape))

    y = np.log1p(train["price"])
    y_mean = np.mean(y)

    # FEATURE EXTRACTION
    pp = pipeline().fit(train, y)
    log('Feature extraction pipeline fitted'.format(time.time() - start_time))
    X = pp.transform(train)
    log('Feature extraction pipeline transformed'.format(time.time() - start_time))

    train_X, train_y = X, y
    valid_X, valid_y = None, None

    if dev_env:
        train_X, valid_X, train_y, valid_y = train_test_split(X, y, shuffle=False,
                                                              test_size=0.05, random_state=100)

    # MODEL CREATION
    log('Fitting model'.format(time.time() - start_time))
    model = FM_FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=0.1, D=train_X.shape[1],
                    alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                    D_fm=200, e_noise=0.0001, iters=15,
                    inv_link="identity", threads=4)

    model.fit(train_X, train_y)
    log('Model created'.format(time.time() - start_time))

    if dev_env:
        valid_y_pred = model.predict(X=valid_X)
        valid_y_pred = np.clip(valid_y_pred, 0, valid_y_pred.max())
        log("Evaluated model score: {}".format(rmsle(np.expm1(valid_y), np.expm1(valid_y_pred))))

    # CREATING SUBMISSION
    def predict():
        for test in pd.read_table('../input/test.tsv', engine='c', chunksize=100000):
            try:
                test_X = pp.transform(test)
                test_y_pred = model.predict(test_X)
                test_y_pred = np.clip(test_y_pred, 0, test_y_pred.max())

                submission = test[['test_id']].copy()
                submission['price'] = np.expm1(test_y_pred)
                yield submission
            except:
                submission_mini_chunks = []
                for i in range(test.shape[0]):
                    test_mini_chunk = test.iloc[i:i + 1]
                    try:
                        test_X_mini_chunk = pp.transform(test_mini_chunk)
                        preds_mini_chunk = model.predict(test_X_mini_chunk)
                        preds_mini_chunk = np.clip(preds_mini_chunk, 0, preds_mini_chunk.max())
                    except:
                        preds_mini_chunk = [y_mean]

                    submission_mini_chunk = test_mini_chunk[['test_id']].copy()
                    submission_mini_chunk['price'] = np.expm1(preds_mini_chunk)
                    submission_mini_chunks.append(submission_mini_chunk)

                yield pd.concat(submission_mini_chunks)

    log('Preparing submission'.format(time.time() - start_time))
    final_submission = pd.concat(predict())
    final_submission.to_csv("bullet_proof_solution.csv", index=False)

    log('Submission saved'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
