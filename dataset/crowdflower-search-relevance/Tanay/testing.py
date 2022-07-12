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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import  pipeline# ,metrics, grid_search
from nltk.stem.porter import PorterStemmer
import re
from bs4 import BeautifulSoup
from sklearn.svm import SVC
from sklearn.feature_extraction import text
import string
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score,classification_report#,confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
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
    train = pd.read_csv("input/train.csv").fillna("")
    test  = pd.read_csv("input/test.csv").fillna("")

    # we dont need ID columns
    idx = test.id.values.astype(int)

    # create labels. drop useless columns
    y = train.median_relevance.values

    def preprocess(x):
        trq=[]        
        s=(" ").join(BeautifulSoup(x).get_text(" ").split(" ")) 
        s=re.sub("[^a-zA-Z]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ") if len(z)>1])
        s=re.sub("[^a-zA-Z]+"," ", s)
        s=" ".join([t for t in s.lower().split(" ") if t not in stop_words and len(t)>1])
        for token in s.split(' '):
            trq.append(token)
        return ' '.join(trq)
    # Fit TFIDF
    import scipy.sparse
    def vectorize(train, tfv_query=None):
        query_data = list(train['query'].apply(preprocess))
        title_data = list(train['product_title'].apply(preprocess))
        if tfv_query is None:
            tfv_query = TfidfVectorizer(min_df=3,  max_features=None,   
                    strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                    ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
                    stop_words =None)

            full_data = query_data + title_data
            tfv_query.fit(full_data)

        return scipy.sparse.hstack([tfv_query.transform(query_data), tfv_query.transform(title_data)]), tfv_query
    
    X, tfv_query = vectorize(train)
    X_test, _ = vectorize(test, tfv_query)
    
    # Initialize SVD
    svd = TruncatedSVD(n_components=400,algorithm='randomized')
    from sklearn.metrics.pairwise import linear_kernel
    class FeatureInserter():
        
        def __init__(self):
            pass
        
        def transform(self, X, y=None):
            distances = []
            quasi_jaccard = []
            print(len(distances), X.shape)
            
            for row in X.tocsr():
                row=row.toarray().ravel()
                cos_distance = linear_kernel(row[:row.shape[0]/2], row[row.shape[0]/2:])
                distances.append(cos_distance[0])
                intersect = row[:row.shape[0]/2].dot(row[row.shape[0]/2:])
                union = (row[:row.shape[0]/2]+row[row.shape[0]/2:]).dot((row[:row.shape[0]/2]+row[row.shape[0]/2:]))
                quasi_jaccard.append(1.0*intersect/union)
                
            print(len(distances), X.shape)
            print(distances[:10])
            
            #X = scipy.sparse.hstack([X, distances])
            return np.matrix([x for x in zip(distances, quasi_jaccard)])
            
        def fit(self, X,y):
            return self
            
        
        def fit_transform(self, X, y, **fit_params):
            self.fit(X,y)
            return self.transform(X)
    
    # Initialize the standard scaler 
    scl = StandardScaler()
    
    # We will use SVM here..
    rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=300,max_depth=16))
    
    # Create the pipeline 
    model = pipeline.Pipeline([('UnionInput', FeatureUnion([('svd', svd), ('dense_features', FeatureInserter())])),('scl', scl),('rf', rf)])

    # Fit Model
    model.fit(X, y)

    preds = model.predict(X_test)
###################################Model performance ############    
    yobs=y
    yhat1=model.predict(X)
    
    print 'Printing model1 performance'
#    print confusion_matrix(yobs, yhat),"\n",
    print accuracy_score(yobs,yhat1),"\n",classification_report(yobs,yhat1)    
    from sklearn import cross_validation
    scores1 = cross_validation.cross_val_score(model, X, y, cv=5,scoring='f1_weighted')
    print round(np.mean(scores1),4),"\n",scores1

############################################################################################ 


    for i in range(len(train.id)):
        s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        s_data.append(s)
        s_labels.append(str(train["median_relevance"][i]))
    for i in range(len(test.id)):
        s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")]) + " " + BeautifulSoup(test.product_description[i]).get_text()
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        t_data.append(s)
    #create sklearn pipeline, fit all, and predit test data
    clf = Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), 
    ('svd', TruncatedSVD(n_components=300, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), 
    ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), 
    ('rf',  OneVsRestClassifier(RandomForestClassifier(n_estimators=300,max_depth=16)))])
    clf.fit(s_data, s_labels)
    t_labels = clf.predict(t_data)

###################################Model performance ############    
    yobs=s_labels
    yhat2=clf.predict(s_data)
    
    print 'Printing model2 performance'
#    print confusion_matrix(yobs, yhat),"\n",
    print accuracy_score(yobs,yhat2),"\n",classification_report(yobs,yhat2)    
#    from sklearn import cross_validation
    scores2 = cross_validation.cross_val_score(clf, s_data, s_labels, cv=5,scoring='f1_weighted')
    print round(np.mean(scores2),4),"\n",scores2

############################################################################################ 
#    Changed n_component from 200 to 300 for model 2 # Result got better by 2%
#    Changed n_component from 400 to 300 for model 1 # Result got worse by 2%
#    Changed n_component from 400 to 450 for model 1 # Result got better by 3.5%
#    Changed n_component from 450 to 500 for model 1 # Result got better by 0.3%
#    Changed n_component from 300 to 400 for model 2 # Result got better to 0.951
#    however cv score is down
#    Changed model2 n_component from 400 to 450# GM 0.956 Model cv scores: 0.651 0.6264
#    Changed model2 C=11 GM 0.956 Model cv scores: 0.6482 0.6218
#    Model2 max_df 500 to 600 GM 0.956 Model cv scores: 0.6504 0.6246
#    Model2 degree 3 to 4 GM 0.954 Model cv scores: 0.6504 0.6233
#    Model2 gamma 0 to 0.001 GM 0.915 Model cv scores: 0.6504 0.6294
#    Model2 algorithm randomized to algorithm='arpack': GM 0.9530  Model cv scores: 0.6484 0.6256
#    450 & 450 # Model1 0.6486 0.9191 model2 0.6241  0.9569 GM 0.9532
#    Model2 onevsrest  0.9255 0.644  0.9605 0.6268 GM 0.9558
#    Model2 OneVsOne 0.9182 0.6488 0.9568 0.6261 GM 0.9522
#    Model2 OutputCode 0.9161 0.6509 0.9553 0.6236 GM 0.9474

############################################################################################
    import math
    p3 = []
    for i in range(len(preds)):
        x =math.sqrt((int(t_labels[i]))*preds[i])
        x = round(x)
        p3.append(int(x))
    ##### Tell the performance of ensemble #######
    yhat2=yhat2.astype(int)
    yobs=np.asarray(yobs,dtype=int)
    yhat=(yhat1+yhat2)/2
    print 'Ensemble by AM',accuracy_score(yobs,yhat),"\n",classification_report(yobs,yhat) 
    yhat=np.sqrt(yhat1*yhat2)
    yhat=yhat.astype(int)
#    yhat = round(yhat)
    print 'Ensemble by GM',accuracy_score(yobs,yhat),"\n",classification_report(yobs,yhat)    
    print 'Model cv scores:',round(np.mean(scores1),4),round(np.mean(scores2),4)
    # Create your first submission file
    submission = pd.DataFrame({"id": idx, "prediction": p3})
    submission.to_csv("ensemble_svc_400_300.csv", index=False)


