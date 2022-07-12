import numpy as np 
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge
from nltk.corpus import stopwords
import string
import gc
import time
import re
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from wordbatch.models import FM_FTRL

def get_rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
        
    def fit(self, x, y=None):
        return self
        
    def transform(self, data_dict):
        return data_dict[self.key]
        
class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.default_preprocessor = CountVectorizer().build_preprocessor()
        
    def fit(self, x, y=None):
        return self
        
    def transform(self, col):
        return col.apply(lambda x: self.default_preprocessor(x))

class DataNormalizer():
    def fit(self, x, y=None):
        return self
        
    def transform(self, col):
        repl = lambda s: s.group(1) + ' carat '
        col = col.str.replace(r'([0-9]+)(\s?(karat|karats|carat|carats|kt?)[\s\.,!])', repl, case=False)
        repl = lambda s: s.group(1) + ' inch '
        col = col.str.replace(r'([0-9]+)("|inch)', repl, case=False)
        repl = lambda s: s.group(1) + ' gb '
        col = col.str.replace(r'([0-9]+)(\s?(gb?|gig))', repl, case=False)
        repl = lambda s: s.group(1) + ' size '
        col = col.str.replace(r'([0-9]+)(\s?sz\.?)', repl, case=False)
        return col
    
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
        
    def split_cat(self, text):
        try: return text.split('/')
        except: return ('missing', 'missing', 'missing')
        
    def transform(self, df):
        features = pd.DataFrame()
        features['name'] = df['name'].fillna('none')
        features['category_name'] = df['category_name'].fillna('none').astype(str)
        features['general_cat'], features['subcat_1'], features['subcat_2'] = zip(*df['category_name'].apply(lambda x: self.split_cat(x)))
        features['brand_name'] = df['brand_name'].fillna('none').astype(str)
        features['shipping'] = df['shipping'].fillna('none').astype(str)
        features['item_description'] = df['item_description'].fillna('none')
        features['item_condition_id'] = df['item_condition_id'].fillna('none').astype(str)
        return features

class TextStats(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
        
    def transform(self, col):
        eng_stopwords = set(stopwords.words('english'))
        return [{'fraction_words_upper': len([w for w in str(text).split() if w.isupper()])/len(str(text).split()),
                'fraction_words_unique': len(set(str(text).split()))/len(str(text).split()),
                'fraction_stopwords': len([w for w in str(text).lower().split() if w in eng_stopwords])/len(str(text).split()),
                'fraction_digits': len([c for c in str(text) if c in string.digits])/len(str(text))} for text in col]
        
vectorizer = Pipeline([
    ('extractfeatures', FeatureExtractor()),
    ('union', FeatureUnion([
        ('name_trigram', Pipeline([
            ('selector', ItemSelector(key='name')),
            ('preprocessor', DataPreprocessor()),
            ('normalizer', DataNormalizer()),
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=75000,
                lowercase=False,
                min_df=2
            ))
        ])),
        ('name_bigram_hash', Pipeline([
            ('selector', ItemSelector(key='name')),
            ('preprocessor', DataPreprocessor()),
            ('normalizer', DataNormalizer()),
            ('hv', HashingVectorizer(
                ngram_range=(1, 2),
                lowercase=False,
                stop_words='english',
                norm='l2'
            ))
        ])),
        ('name_stats', Pipeline([
            ('selector', ItemSelector(key='name')),
            ('stats', TextStats()),
            ('vect', DictVectorizer())
        ])),
        ('category_name', Pipeline([
            ('selector', ItemSelector(key='category_name')),
            ('preprocessor', DataPreprocessor()),
            ('cv', CountVectorizer(
                token_pattern='.+',
                
            ))
        ])),
        ('general_cat', Pipeline([
            ('selector', ItemSelector(key='general_cat')),
            ('preprocessor', DataPreprocessor()),
            ('cv', CountVectorizer(
                token_pattern='.+',
                min_df=2
            ))
        ])),
        ('subcat_1', Pipeline([
            ('selector', ItemSelector(key='subcat_1')),
            ('preprocessor', DataPreprocessor()),
            ('cv', CountVectorizer(
                token_pattern='.+',
                min_df=2
            ))
        ])),
        ('subcat_2', Pipeline([
            ('selector', ItemSelector(key='subcat_2')),
            ('preprocessor', DataPreprocessor()),
            ('cv', CountVectorizer(
                token_pattern='.+',
                min_df=2
            ))
        ])),
        ('brand_name', Pipeline([
            ('selector', ItemSelector(key='brand_name')),
            ('preprocessor', DataPreprocessor()),
            ('cv', CountVectorizer(
                token_pattern='.+',
                min_df=2
            ))
        ])),
        ('shipping', Pipeline([
            ('selector', ItemSelector(key='shipping')),
            ('preprocessor', DataPreprocessor()),
            ('cv', CountVectorizer(
                token_pattern='\d+'
            ))
        ])),
        ('item_condition_id', Pipeline([
            ('selector', ItemSelector(key='item_condition_id')),
            ('preprocessor', DataPreprocessor()),
            ('cv', CountVectorizer(
                token_pattern='\d+'
            ))
        ])),
        ('item_description_trigram', Pipeline([
            ('selector', ItemSelector(key='item_description')),
            ('preprocessor', DataPreprocessor()),
            ('normalizer', DataNormalizer()),
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=100000,
                lowercase=False
            ))
        ])),
        ('item_description_bigram_hash', Pipeline([
            ('selector', ItemSelector(key='item_description')),
            ('preprocessor', DataPreprocessor()),
            ('normalizer', DataNormalizer()),
            ('hv', HashingVectorizer(
                ngram_range=(1, 2),
                lowercase=False,
                stop_words='english',
                norm='l2'
            ))
        ])),
        ('item_description_stats', Pipeline([
            ('selector', ItemSelector(key='item_description')),
            ('stats', TextStats()),
            ('vect', DictVectorizer())
        ])),
    ]))
])

def main():
    start_time = time.time()
    
    train = pd.read_table('../input/train.tsv', engine='c')
    y_train = np.log1p(train['price'])

    print('[{}] Start transforming train data.'.format(time.time() - start_time))
    X_train = vectorizer.fit_transform(train)
    print(X_train.shape)
    print('[{}] Finished transforming train data.'.format(time.time() - start_time))
    
    del train
    gc.collect()
    
    print('[{}] Start training ridge model.'.format(time.time() - start_time))
    modelR = Ridge(
        solver='auto',
        fit_intercept=True,
        alpha=3,
        normalize=False,
        max_iter=100,
        tol=0.01
    )
    modelR.fit(X_train, y_train)
    print('[{}] Finished training ridge model.'.format(time.time() - start_time))
    
    modelF = FM_FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=0.1, D=X_train.shape[1],
                    alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                    D_fm=200, e_noise=0.0001, iters=15,
                    inv_link="identity", threads=4)

    modelF.fit(X_train, y_train)
    
    print('[{}] Finished training model.'.format(time.time() - start_time))
    
    def load_test():
        for df in pd.read_csv('../input/test.tsv', sep='\t', chunksize=350000, engine='c'):
            yield df
    
    predsL = []
    predsF = []
    predsR = []
    test_ids = []
    
    print('[{}] Transform test data.'.format(time.time() - start_time))
    for X_test in load_test():
        test_ids = np.append(test_ids, X_test['test_id'])
        X_test = vectorizer.transform(X_test)
        print('[{}] Make batch predictions.'.format(time.time() - start_time))
        batch_predsR = modelR.predict(X_test)
        predsR = np.append(predsR, batch_predsR)
        batch_predsF = modelF.predict(X_test)
        predsF = np.append(predsF, batch_predsF)
        
    preds = (0.5*predsR + 0.5*predsF)
    preds = np.expm1(preds)
    print('[{}] Predictions completed.'.format(time.time() - start_time))

    submission = pd.DataFrame({
        "test_id": test_ids.astype(int),
        "price": preds
    })
    submission.to_csv('submission.csv', index=False, columns=['test_id', 'price'])

if __name__ == '__main__':
    main()