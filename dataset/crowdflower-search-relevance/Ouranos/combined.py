
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler    
from sklearn.svm import SVC  
from sklearn.cross_validation import KFold 
from nltk.stem.porter import PorterStemmer
import re
from bs4 import BeautifulSoup  
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import text
import string

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

# Use Pandas to read in the training and test data
train = pd.read_csv("../input/train.csv").fillna("")
test  = pd.read_csv("../input/test.csv").fillna("")
idx = test.id.values.astype(int)

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
 
def Porter_SVM():  

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
    def vectorize(train, tfv_query=None):
        query_data = list(train['query'].apply(preprocess))
        title_data = list(train['product_title'].apply(preprocess))
        if tfv_query is None:
            tfv_query = TfidfVectorizer(min_df=3,  max_features=None,   
                    strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                    ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
                    stop_words =stop_words)

            full_data = query_data + title_data
            np.random.seed(125) 
            tfv_query.fit(full_data)

        return scipy.sparse.hstack([tfv_query.transform(query_data), tfv_query.transform(title_data)]), tfv_query
    
    X, tfv_query = vectorize(train)
    X_test, _ = vectorize(test, tfv_query)
    
    # Initialize SVD
    #seednum = 125
    np.random.seed(125)    
    svd = TruncatedSVD(n_components=400)
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
    svm_model = SVC(C=10.)
    
    # Create the pipeline 
    np.random.seed(125)
    model = Pipeline([('UnionInput', FeatureUnion([('svd', svd), ('dense_features', FeatureInserter())])),
    						 ('scl', scl),
                    	     ('svm', svm_model)])
    # Fit Model
    np.random.seed(125)
    model.fit(X, y)

    preds = model.predict(X_test)

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
    clf = Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), 
    ('svd', TruncatedSVD(n_components=300, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), 
    ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), 
    ('svm', SVC(C=10.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
    np.random.seed(125)
    clf.fit(s_data, s_labels)
    t_labels = clf.predict(t_data)
    
    return np.array(t_labels), np.array(preds)
    
    
 

def LR_features(X, titles, queries, description):

    u = np.unique(queries)
   
    O = np.zeros((len(titles), len(u)))

    for i in range(len(u)):
        q = u[i] 
        titles = np.array(titles)
        queries = np.array(queries)
        train_idx = list(np.where(queries[:10158]==q)[0])
        test_idx = list(np.where(queries[10158:]==q)[0] + 10158)
        all_idx = train_idx + test_idx  
        np.random.seed(125) 
        vect = CountVectorizer(binary=True).fit(queries[train_idx])

        V = vect.transform(titles[all_idx]).toarray() 
   
          
        O[all_idx,i] = V.mean(1).ravel() 
 
    V1=sparse.csr_matrix(scale(O)) 
    
    O = np.zeros((len(titles), len(u)))

    docs = np.array(["%s %s"%(a,b) for a,b in zip(titles,queries)])

    for i in range(len(u)):
        q = u[i] 
        titles = np.array(titles)
        queries = np.array(queries)
        train_idx = list(np.where(queries[:10158]==q)[0])
        test_idx = list(np.where(queries[10158:]==q)[0] + 10158)
        all_idx = train_idx + test_idx  
        np.random.seed(125) 
        vect = CountVectorizer(binary=True).fit(docs[train_idx])

        #V = vect.transform(titles[all_idx]).toarray()
        V = vect.transform(docs[all_idx])
 

        A = TruncatedSVD(1).fit(V[:len(train_idx)]).transform(V) 
 
          
        O[all_idx,i] = A.ravel()

    W=scale(O)
    W=sparse.hstack([V1,W]).tocsr() 
    return W
    

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
    rater_a = y
    rater_b = np.round(y_pred)
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
 
def scale(X):
    return StandardScaler().fit(X[:10158]).transform(X) 
    
def percent_features(titles, queries, description):
    X = np.zeros((len(titles), 3))  
    titles = np.array(titles)
    queries = np.array(queries)
    i = 0
    for q in np.unique(queries): 
        train_idx = list(np.where(queries[:10158]==q)[0])
        test_idx = list(np.where(queries[10158:]==q)[0] + 10158)
        all_indices = train_idx + test_idx   
        np.random.seed(125) 
        vect = CountVectorizer(binary=True).fit(queries[train_idx]) 
        
        X[all_indices,0] = i
        X[all_indices,1] = vect.transform(titles[all_indices]).toarray().mean(1).ravel() 
        X[all_indices,2] = vect.transform(description[all_indices]).toarray().mean(1).ravel()  
        
        i += 1 
        
    X[:,1] = scale(X[:,1])
    return X 
    
def SVD_features(titles, queries):  
    vect =  CountVectorizer(binary=True)
    np.random.seed(125)
    CSR = vect.fit(titles[:10158]).transform(titles) 
 
    X = np.zeros((len(queries), 3)) 
    for i in np.unique(queries): 
        idx = np.where(queries==i)[0] 
        feats = np.unique(CSR[idx].nonzero()[1])
        X[idx] = TruncatedSVD(3).fit_transform(CSR[idx][:,feats])
    X[:,1] = scale(X[:,1])
    return X[:,[0,1,2]]
   
def cv(m,X,y): 
    e = []
    for train, test in KFold(len(y), 3):
        p = m.fit(X[train],y[train]).predict(X[test])
        try:
            e.append(quadratic_weighted_kappa(y[test],p))
        except:
            pass
    return np.mean(e)
   
def get_data():
    def stem(x): 
        stemmer = PorterStemmer()
        res = []
        for d in x:
            s=(" ").join([z for z in BeautifulSoup(d).get_text(" ").split(" ")]) 
            s=re.sub("[^a-zA-Z0-9]"," ", s)
            s=[stemmer.stem(z) for z in s.split(" ")]
            res.append(s) 
        return res
    
    def clean(x):
        stemmer = PorterStemmer()
        html_free = BeautifulSoup(x).get_text(" ").lower()
        cleaned = re.sub("[ ]+", " ", re.sub("[^a-zA-Z0-9]"," ", html_free)) 
        res = ""
        for z in cleaned.split(" "):
            res = res + " " + stemmer.stem(z)
        return res.strip() 
    
    titles_train =  [clean(a) for a in  train["product_description"].values]
    titles_test = [clean(a) for a in test["product_description"].values] 

    titles_train =  [clean(a) for a in  train["product_title"].values]
    titles_test = [clean(a) for a in test["product_title"].values] 

    query_train = [clean(a) for a in  train["query"].values]
    query_test = [clean(a) for a in test["query"].values]

    description = np.array(titles_train + titles_test) 
    titles = np.array(titles_train + titles_test)
    queries = np.array(query_train + query_test)
    
    docs = ["%s %s"%(a,b) for a,b in zip(titles,queries)]
    np.random.seed(125)
    CSR = TfidfVectorizer(min_df=5,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english').fit(docs[:10158]).transform(docs)
    
    y = train.median_relevance.values
    
    P = percent_features(titles, queries, description)
    SVD = SVD_features(titles, queries)
    
    X = np.column_stack([P,SVD])
    
    V = LR_features(CSR,titles,queries,description)
    V = sparse.hstack([CSR,V]).tocsr()
    return X , V, y
    
def SVC_LR_models():
    X, V, y = get_data()
    np.random.seed(125) 
    svc = SVC(random_state=0, C=5)
    p1 = svc.fit(X[:10158],y).predict(X[10158:])
    
    print(X.shape)
    #print(cv(svc,X,y))
    np.random.seed(125) 
    lr = LogisticRegression(class_weight="auto",random_state=2, C=12)
    p2 = lr.fit(V[:10158],y).predict(V[10158:])
    return p1, p2
 
if __name__=="__main__":
    #p1,p2 = Porter_SVM()
    p3,p4 = SVC_LR_models()
    pred = np.column_stack([p3,p4]).prod(1)**0.5
    pred = np.round(pred).astype("i") 
        
    # Create your first submission file
    submission = pd.DataFrame({"id": idx, "prediction": pred})
    submission.to_csv(str("3_SVM_1_LR_ensemble"+str(125)+".csv"), index=False)