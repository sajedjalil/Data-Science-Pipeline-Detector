 
"""
Beating the Benchmark 
Search Results Relevance @ Kaggle
__author__ : Abhishek

multiprocessing support : Antonis Nikitakis

"""
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


def load_preds(csv_file):
    preds = pd.read_csv(csv_file)
    preds=preds.drop('id', axis=1)
    preds=preds['prediction']
    preds=list (preds)
    return preds

if __name__ == '__main__':
    
    def model1(n_jobs=2,compute_csv=True):
        
        if (compute_csv):        
            #--------------------------Put your model here-----------------------------    
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
            X_test = tfv.transform(testdata)
            
            # Initialize SVD
            svd = TruncatedSVD()
            
            # Initialize the standard scaler 
            scl = StandardScaler()
            
            # We will use SVM here..
            svm_model = SVC()
            
            # Create the pipeline 
            clf = pipeline.Pipeline([('svd', svd),
            						 ('scl', scl),
                            	     ('svm', svm_model)])
            
            # Create a parameter grid to search for best parameters for everything in the pipeline
            param_grid = {'svd__n_components' : [340],
                          'svm__C': [16]}
            
            # Kappa Scorer 
            kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)
            
            # Initialize Grid Search Model
            model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=kappa_scorer,
                                             verbose=10, n_jobs=n_jobs, iid=True, refit=True, cv=2)
                                             
            # Fit Grid Search Model
            model.fit(X, y)
            print("Best score: %0.3f" % model.best_score_)
            print("Best parameters set:")
            best_parameters = model.best_estimator_.get_params()
            for param_name in sorted(param_grid.keys()):
            	print("\t%s: %r" % (param_name, best_parameters[param_name]))
            
            # Get best model
            best_model = model.best_estimator_
            
            # Fit model with best parameters optimized for quadratic_weighted_kappa
            best_model.fit(X,y)
            preds = best_model.predict(X_test)
            #-------------------------------------------------------

            # Create partial submission file
            test  = pd.read_csv("../input/test.csv").fillna("") 
            idx = test.id.values.astype(int)
            submission = pd.DataFrame({"id": idx, "prediction": preds})
            submission.to_csv("./model1.csv", index=False)  
        
        else:
            preds=load_preds("./model1.csv")          

     
        return preds
    
    
    def model2(n_jobs=2,compute_csv=True):

        
        if (compute_csv):

            #----------------------Put your model here------------------------------    
            #load data
            train = pd.read_csv("../input/train.csv").fillna("")
            test  = pd.read_csv("../input/test.csv").fillna("")             
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
            ('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), 
            ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), 
            ('svm', SVC(C=10.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
            clf.fit(s_data, s_labels)
            t_labels = clf.predict(t_data)
            #-------------------------------------------------------
            
            # Create partial submission file
            test  = pd.read_csv("../input/test.csv").fillna("")             
            idx = test.id.values.astype(int)
            submission = pd.DataFrame({"id": idx, "prediction": t_labels})
            submission.to_csv("model2.csv", index=False)
        else:
            t_labels=load_preds("./model2.csv")
            
        return t_labels

        
    import multiprocessing
    import time


    def start_process():
        print ('Starting in multi processing', multiprocessing.current_process().name)
    
    procs={
        0:model1,
        1:model2
       } 
 

    ids=[0,1]
    

    def worker(ids):
        out=procs[ids]()
        
        return out
            
    start = time.time()

    
    pool_size =multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=pool_size,initializer=start_process, maxtasksperchild=2)
    pool_outputs = pool.map(worker, ids)
    pool.close()
    pool.join()    
    
    end = time.time()
    print ('time=%f'%(end - start) )
  
      
    


        
    import math
    p3 = []
    for i in range(len(pool_outputs[1])):
        x = int(pool_outputs[0][i])*0.5 + int(pool_outputs[1][i])*0.5
        x = math.floor(x)
        p3.append(int(x))
        
        
    
    # p3 = (t_labels + preds)/2
    # p3 = p3.apply(lambda x:math.floor(x))
    # p3 = p3.apply(lambda x:int(x))
    
    # preds12 = 

    # Create your first submission file
    test  = pd.read_csv("../input/test.csv").fillna("") 
    idx = test.id.values.astype(int)
    submission = pd.DataFrame({"id": idx, "prediction": p3})
    submission.to_csv("beating_the_benchmark_05_05.csv", index=False)



