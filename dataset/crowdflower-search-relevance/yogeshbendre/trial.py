import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import decomposition, pipeline, metrics, grid_search

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

#load data
train = pd.read_csv("../input/train.csv").fillna("")
test  = pd.read_csv("../input/test.csv").fillna("")

#remove html, remove non text or numeric, make query and title unique features for counts using prefix (accounted for in stopwords tweak)

#Snow Ball Stemmer
stemmer = SnowballStemmer("porter", ignore_stopwords=False)
SBstemmer = SnowballStemmer("english", ignore_stopwords=False)

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
    s= (" ").join([SBstemmer.stem(z) for z in s.split(" ")])
    #s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
    s_data.append(s)
    s_labels.append(str(train["median_relevance"][i]))

for i in range(len(test.id)):
    s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")]) + " " + BeautifulSoup(test.product_description[i]).get_text()
    s=re.sub("[^a-zA-Z0-9]"," ", s)
    s= (" ").join([SBstemmer.stem(z) for z in s.split(" ")])
    #s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
    t_data.append(s)
    

# the infamous tfidf vectorizer (Do you remember this one?)
tfv = TfidfVectorizer(min_df=3,max_df=500, max_features=None,strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = 'english')
    
# Initialize SVD
svd = TruncatedSVD()
    
# Initialize the standard scaler 
scl = StandardScaler()
sminmax = MinMaxScaler()    
# We will use SVM here..
svm_model = SVC()
    
# Create the pipeline 
clf = pipeline.Pipeline([('tfv',tfv),('svd', svd),('scl', scl),('sminmax',sminmax),('svm', svm_model)])
clf2 = pipeline.Pipeline([('tfv',tfv),('svd', svd),('scl', scl),('sminmax',sminmax)])
# Create a parameter grid to search for best parameters for everything in the pipeline
param_grid = {'svd__n_components' : [200],'svm__C': [20],'svm__gamma':[4],'svm__kernel':['rbf'],'svm__degree':[3],'svm__coef0':[1]}
param_grid2 = {'svd__n_components' : [200]}

clf2.fit_transform(s_data,s_labels)
s_data2=clf2.transform(s_data)
t_data2=clf2.transform(t_data)

print("Sample Data")
print(len(s_data2))
print(s_data2.shape)
print(s_data2[3])
print(s_data2[6])

# Kappa Scorer 
kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)

# Initialize Grid Search Model
model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=kappa_scorer,verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)
                                     
# Fit Grid Search Model
model.fit(s_data,s_labels)
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
# Get best model
best_model = model.best_estimator_
    
# Fit model with best parameters optimized for quadratic_weighted_kappa
best_model.fit(s_data,s_labels)
t_labels = best_model.predict(t_data)
print(best_model)
#output results for submission
with open("submission_with_sb_stemmer.csv","w") as f:
    f.write("id,prediction\n")
    for i in range(len(t_labels)):
        f.write(str(test.id[i])+","+str(t_labels[i])+"\n")
f.close()


