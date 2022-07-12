from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

DATA_PATH = Path('../input')
random_seed = 17

train_df = pd.read_csv(DATA_PATH/'train.csv')
test_df = pd.read_csv(DATA_PATH/'test.csv')

train_df['class'] = np.where(train_df['target'] >= 0.5, 1, 0)

class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.columns]

class NBTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y=None):
        y = y.values
        
        pos_count = X[y==1].sum(0) 
        neg_count = X[y==0].sum(0)
        n = X.shape[1]
        p = (pos_count + self.alpha) / (pos_count.sum() + self.alpha * n)
        q = (neg_count + self.alpha) / (neg_count.sum() + self.alpha * n)
        self.r_ = np.log(p / q)
        return self
    
    def transform(self, X, y=None):
        return X.multiply(self.r_)

class TfidfVectorizerPlus(TfidfVectorizer):
    def __init__(self, fit_add=None, norm_type=None, pivot=5, slope=0.2, 
                       input='content', encoding='utf-8', decode_error='strict', 
                       strip_accents=None, lowercase=True, preprocessor=None, 
                       tokenizer=None, analyzer='word', stop_words=None, 
                       token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1), 
                       max_df=1.0, min_df=1, max_features=None, vocabulary=None, 
                       binary=False, dtype=np.float64, norm='l2', 
                       use_idf=True, smooth_idf=True, sublinear_tf=False):
        super().__init__(input, encoding, decode_error,
                         strip_accents, lowercase, preprocessor,
                         tokenizer, analyzer, stop_words,
                         token_pattern, ngram_range,
                         max_df, min_df, max_features, vocabulary,
                         binary, dtype, norm,
                         use_idf, smooth_idf, sublinear_tf)
        
        self.fit_add = fit_add
        self.norm_type = norm_type
        self.pivot = pivot
        self.slope = slope
    
    def fit(self, X, y=None):
        if self.fit_add is not None:
            X_new = pd.concat([X, self.fit_add])
        else:
            X_new = X
        
        super().fit(X_new, y)
        return self
        
    def transform(self, X, y=None):
        res = super().transform(X)
            
        if self.norm_type == 'pivot_cosine':
            norm_factor = (1 - self.slope) * self.pivot + self.slope * sparse.linalg.norm(res, axis=1).reshape(-1, 1)
            res = csr_matrix(res.multiply(1 / norm_factor))
        elif self.norm_type == 'pivot_unique':
            unique_terms_num = (res > 0).sum(axis=1)
            norm_factor = (1 - self.slope) * self.pivot + self.slope * unique_terms_num
            res = csr_matrix(res.multiply(1 / norm_factor))
        elif self.norm_type is not None:
            raise ValueError('Incorrect normalization type')
            
        return res
        
        
pipe = Pipeline([
    ('extract', ColumnExtractor(columns='comment_text')),
    ('vec', TfidfVectorizerPlus()),
    ('nb_features', NBTransformer()),
    ('clf', LinearSVC())
])

params = {              
    'vec': TfidfVectorizerPlus(),
    'vec__strip_accents': 'unicode', 
    'vec__lowercase': True,
    'vec__stop_words': None, 
    'vec__token_pattern': r'\b\w+\b', 
    'vec__ngram_range': (1, 1), 
    'vec__max_df': 0.8,
    'vec__min_df': 2, 
    'vec__max_features': None, 
    'vec__binary': False, 
    'vec__norm': 'l2',
    'vec__use_idf': True, 
    'vec__smooth_idf': True, 
    'vec__sublinear_tf': True, 
    
    'vec__fit_add': test_df['comment_text'],
    'vec__norm_type': 'pivot_unique',
    'vec__pivot': 30,
    'vec__slope': 0.2,

    'nb_features__alpha': 0.2,
              
    'clf__penalty': 'l2',
    'clf__loss': 'squared_hinge',
    'clf__dual': False,
    'clf__C': 0.1, 
    'clf__class_weight': None,
    'clf__random_state': random_seed
}

pipe.set_params(**params)
pipe.fit(train_df, train_df['class'])

submission_df = pd.read_csv(DATA_PATH/'sample_submission.csv', index_col='id')
y_margins = pipe.decision_function(test_df)
y_val_pred = (y_margins - y_margins.min()) / (y_margins.max() - y_margins.min())
submission_df['prediction'] = y_val_pred
submission_df.to_csv('submission.csv')