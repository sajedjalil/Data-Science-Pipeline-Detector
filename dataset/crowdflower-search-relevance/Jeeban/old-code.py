import difflib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search
from nltk.stem.porter import *
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
# array declarations
sw=[]
s_data = []
s_labels = []
t_data = []
t_labels = []
#stopwords tweak - more overhead
stop_words = ['http','www','img','border','0','1','2','3','4','5','6','7','8','9']
stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)
for stw in stop_words:
    sw.append("q"+stw)
    sw.append("z"+stw)
stop_words = text.ENGLISH_STOP_WORDS.union(sw)

def fuzy_match_score(a,b):
    seq=difflib.SequenceMatcher(a=a.lower(), b=b.lower())
    score=seq.ratio()
    return(score)

def matched_words(document_1_text,document_2_text):
    document_1_words_lower=document_1_text.lower()
    document_2_words_lower=document_2_text.lower()
    document_1_words = document_1_words_lower.split()
    document_2_words = document_2_words_lower.split()
    common = set(document_1_words).intersection( set(document_2_words) )
    matched_words=len(common)
    #return(common_words)
    return(matched_words)

def matched_words_percent(document_1_text,document_2_text):
    document_1_words_lower=document_1_text.lower()
    document_2_words_lower=document_2_text.lower()
    document_1_words = document_1_words_lower.split()
    document_2_words = document_2_words_lower.split()
    common = set(document_1_words).intersection( set(document_2_words) )
    matched_words=100*len(common)/len(document_1_words)
    #return(common_words)
    return(matched_words)

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
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    
    # we dont need ID columns
    idx = test.id.values.astype(int)
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)
    
    # create labels. drop useless columns
    y = train.median_relevance.values
    train = train.drop(['median_relevance', 'relevance_variance'], axis=1)
    
    # do some lambda magic on text columns
    traindata = list(train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    testdata = list(test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    
    # the infamous tfidf vectorizer (Do you remember this one?)
    tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
    
    # Fit TFIDF
    tfv.fit(traindata)
    X =  tfv.transform(traindata) 
    #X_test = tfv.transform(testdata)
    train['q_pt_match_score']=np.vectorize(fuzy_match_score)(train['query'], train['product_title'])
    train['q_pd_match_score']=np.vectorize(fuzy_match_score)(train['query'], train['product_description'])
    #train['q_pt_match_cnt_percet']=np.vectorize(matched_words_percent)(train['query'], train['product_title'])
    #train['q_pd_match_cnt_percet']=np.vectorize(matched_words_percent)(train['query'], train['product_description'])
    #train['q_pt_match_cnt']=np.vectorize(matched_words)(train['query'], train['product_title'])
    
    
    
    
    # Initialize SVD
    #svd = TruncatedSVD()
    
    # Initialize the standard scaler 
    scl = StandardScaler()
    
    # We will use SVM here..
    svm_model = SVC()
    
    # Create the pipeline 
    clf = pipeline.Pipeline([
    						 ('scl', scl),
                    	     ('svm', svm_model)])
    
    # Create a parameter grid to search for best parameters for everything in the pipeline
    param_grid = {
                  'svm__C': [10]}
    
    # Kappa Scorer 
    #kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)
    
    # Initialize Grid Search Model
    #model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=kappa_scorer,
     #                                verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)
                                     
    # Fit Grid Search Model
    #model.fit(X, y)
    #print("Best score: %0.3f" % model.best_score_)
    #train_data=pd.DataFrame(X)
    
    #len(X.columns))
    #print(X[1:5])
    #print(train[1:10])