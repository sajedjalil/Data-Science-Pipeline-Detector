"""
Original:
Beating the Benchmark 
Search Results Relevance @ Kaggle
__author__ : Abhishek

Forked from Clubbing2Benchmark 
__author__: Justfor

changed by
__author__: Henning Sperr

Modified by justfor
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search
from nltk.stem.porter import *
import re
import time
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction import text
import string
from sklearn.pipeline import Pipeline, FeatureUnion

# array declarations
sw=[]
s_data = []
s_labels = []
t_data = []
t_labels = []
stemmer = PorterStemmer()
#stopwords tweak - more overhead
stop_words = ['http','www','img','border','color','style','padding','table','font','thi','inch','ha','width','height',
'0','1','2','3','4','5','6','7','8','9']
#stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)

#stop_words = ['http','www','img','border','0','1','2','3','4','5','6','7','8','9']
stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)

punct = string.punctuation
punct_re = re.compile('[{}]'.format(re.escape(punct)))

#remove html, remove non text or numeric, make query and title unique features for counts using prefix (accounted for in stopwords tweak)
stemmer = PorterStemmer()
## Stemming functionality
class stemmerUtility(object):
    """Stemming functionality"""
    @staticmethod
    def stemPorter(review_text):
        porter = PorterStemmer()
        preprocessed_docs = []
        for doc in review_text:
            final_doc = []
            for word in doc:
                final_doc.append(porter.stem(word))
                #final_doc.append(wordnet.lemmatize(word)) #note that lemmatize() can also takes part of speech as an argument!
            preprocessed_docs.append(final_doc)
        return preprocessed_docs

# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


if __name__ == '__main__':

    # Load the training file
    train = pd.read_csv("../input/train.csv").fillna("")
    test  = pd.read_csv("../input/test.csv").fillna("")
    #train=train.head(4000)
    #test=test.head(5000)

    # we dont need ID columns
    idx = test.id.values.astype(int)

    # create labels. drop useless columns
    y = train.median_relevance.values

    def preprocess(x):
        x=x.lower()
        x=punct_re.sub(' ', x)
        new_x = []
        for token in x.split(' '):
            new_x.append(stemmer.stem(token))
        return ' '.join(new_x)
    # Fit TFIDF
    import scipy.sparse
    def vectorize(train, tfv_query=None, include_description=False):
        query_data = list(train['query'].apply(preprocess))
        title_data = list(train['product_title'].apply(preprocess))
        if include_description:
             desc_data = list(train['product_description'].apply(preprocess))
        
        if tfv_query is None:
            tfv_query = TfidfVectorizer(min_df=3,  max_features=None,   
                    strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                    ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
                    stop_words =stop_words)
            
            if include_description:
                full_data = query_data + title_data + desc_data
            else:
                full_data = query_data + title_data
            tfv_query.fit(full_data)

        if include_description:
            data = scipy.sparse.hstack([tfv_query.transform(query_data), tfv_query.transform(title_data)])
            return scipy.sparse.hstack([data, tfv_query.transform(desc_data)]), tfv_query
        else:
            return scipy.sparse.hstack([tfv_query.transform(query_data), tfv_query.transform(title_data)]), tfv_query
    
    
    t0=time.time()
    X, tfv_query = vectorize(train)
    X_test, _ = vectorize(test, tfv_query)
    print('Vectorize took: {}'.format(time.time()-t0))
    # Initialize SVD
    svd = TruncatedSVD(n_components=400)
    from sklearn.metrics.pairwise import linear_kernel
    class FeatureInserter():
        
        def __init__(self):
            pass
        
        def transform(self, X, y=None):
            distances = []
            unigram_and_bigram_hits = []
            quasi_jaccard = []
            print(len(distances), X.shape)
            
            for row in X.tocsr():
                row=row.toarray().ravel()
                cos_distance = linear_kernel(row[:row.shape[0]/2], row[row.shape[0]/2:])
                distances.append(cos_distance[0])
                intersect = row[:row.shape[0]/2].dot(row[row.shape[0]/2:])
                union = (row[:row.shape[0]/2]+row[row.shape[0]/2:]).dot((row[:row.shape[0]/2]+row[row.shape[0]/2:]))
                quasi_jaccard.append(1.0*intersect/union)
                unigram_and_bigram_hits.append(intersect)
            print(len(distances), X.shape)
            print(distances[:10])
            
            #X = scipy.sparse.hstack([X, distances])
            return np.matrix([x for x in zip(unigram_and_bigram_hits, distances, quasi_jaccard)])
            
        def fit(self, X,y):
            return self
            
        
        def fit_transform(self, X, y, **fit_params):
            self.fit(X,y)
            return self.transform(X)
    
    # Initialize the standard scaler 
    scl = StandardScaler()
    
    # We will use SVM here..
    svm_model = SVC(C=10.)
    
    # Create the pipeline 
    svm_pipeline = pipeline.Pipeline([('UnionInput', FeatureUnion([('svd', svd), ('dense_features', FeatureInserter())])),
    						 ('scl', scl),
                    	     ('svm', svm_model)])
    t0 = time.time()
    # Fit Model
    svm_pipeline.fit(X, y)
    print('Fitting SVM took: {}'.format(time.time()-t0))
    
    # We will use SVM here..
    logistic_model = LogisticRegression(C=10., class_weight='auto')
    
    # Create the pipeline 
    logistic_pipeline = pipeline.Pipeline([('UnionInput', FeatureUnion([('svd', svd), ('dense_features', FeatureInserter())])),
    						 ('scl', scl),
                    	     ('svm', logistic_model)])
    # Fit Model
    t0 = time.time()
    logistic_pipeline.fit(X, y)
    print('Fitting LogReg took: {}'.format(time.time()-t0))
    
    svm_preds = svm_pipeline.predict(X_test)
    logistic_preds = logistic_pipeline.predict(X_test)

    t0 = time.time()
    X, tfv_query = vectorize(train, include_description=True)
    X_test, _ = vectorize(test, tfv_query, include_description=True)

    print('Prepro last model took: {}'.format(time.time()-t0))
    t0 = time.time()
    #create sklearn pipeline, fit all, and predit test data
    clf = Pipeline([#('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), 
    ('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), 
    ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), 
    ('svm', SVC(C=10.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
    clf.fit(X, y)
    t_labels = clf.predict(X_test)
    print('Fitting Last SVM took: {}'.format(time.time()-t0))
    
    
    final_preds = (t_labels+svm_preds+logistic_preds)/3.
    final_preds = np.floor(final_preds).astype(int)
        
    # Create your first submission file
    submission = pd.DataFrame({"id": idx, "prediction": final_preds})
    submission.to_csv("ensemble_svc_different_preproc.csv", index=False)


