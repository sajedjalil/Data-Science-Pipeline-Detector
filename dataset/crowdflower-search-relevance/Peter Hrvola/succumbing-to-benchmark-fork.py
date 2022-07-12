from nltk.stem.porter import *
import pandas as pd
import numpy as np
import re
import unicodedata
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics import accuracy_score, mean_absolute_error
# array declarations
sw=[]
s_data = []
s_labels = []
t_data = []
t_labels = []
cs_labels = []
stemmer = PorterStemmer()
#stopwords tweak - more overhead
stop_words1 = ['http','www','img','border','color','style','padding','table','font']
stop_words1 = list(text.ENGLISH_STOP_WORDS.union(stop_words1))
for stw in stop_words1:
    sw.append("q"+stw)
    sw.append("z"+stw)
for i in range(len(stop_words1)):
    stop_words1[i]=stemmer.stem(stop_words1[i])
stop_words1 = list(text.ENGLISH_STOP_WORDS.union(sw))
#load data
train = pd.read_csv("../input/train.csv").fillna("")
test  = pd.read_csv("../input/test.csv").fillna("")
#remove html, remove non text or numeric, stem, make query and title unique features for counts using prefix (accounted for in stopwords tweak)
for i in range(len(train.id)):
    tx = BeautifulSoup(train.product_description[i])    
    tx1 = [x.extract() for x in tx.findAll('script')]
    tx = tx.get_text(" ").strip().lower()
    if "translation tool" in tx:
        tx = tx[-500:]
    s=(" ").join(["q"+ z for z in train["query"][i].split(" ")]) + " " + (" ").join(["z"+ z for z in train.product_title[i].split(" ")]) + " " + tx
    s=re.sub("[^a-zA-Z0-9]"," ", s)
    s= re.sub("[0-9]{1,3}px"," ", s)
    s= re.sub(" [0-9]{1,6} |000"," ", s)
    s= (" ").join([stemmer.stem(z) for z in s.split(" ") if len(z)>2])
    s_data.append(s)
    s_labels.append(str(train["median_relevance"][i]))
for i in range(len(test.id)):
    tx = BeautifulSoup(test.product_description[i])
    tx1 = [x.extract() for x in tx.findAll('script')]
    tx = tx.get_text(" ").strip().lower()
    tx = (" ").join([z for z in tx.split(" ")])
    s=(" ").join(["q"+ z for z in test["query"][i].split(" ")]) + " " + (" ").join(["z"+ z for z in test.product_title[i].split(" ")]) + " " + tx
    s=re.sub("[^a-zA-Z0-9]"," ", s)
    s= re.sub("[0-9]{1,3}px"," ", s)
    s= re.sub(" [0-9]{1,6} |000"," ", s)
    s= (" ").join([stemmer.stem(z) for z in s.split(" ") if len(z)>2])
    t_data.append(s)
#create sklearn pipeline, fit all, and predit test data
clf = Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = stop_words1)), ('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), ('svm', SVC(C=10.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.0001, cache_size=300, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
clf.fit(s_data, s_labels)
t_labels = clf.predict(t_data)
cs_labels = clf.predict(s_data)
#output results for submission
with open("submission.csv","w") as f:
    f.write("id,prediction\n")
    for i in range(len(t_labels)):
        f.write(str(test.id[i])+","+str(t_labels[i])+"\n")
f.close()

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

# kappa
print(quadratic_weighted_kappa(s_labels,cs_labels))
print(confusion_matrix(s_labels,cs_labels))
#for further review and improvement
#print("Feature Names ---------------------")
#print(clf.named_steps['v'].get_feature_names())
#print("Parameters ---------------------")
#print(clf.named_steps['v'].get_params(deep=True))
#print("Stop Words List ---------------------")
#print(stop_words1)
print(s_data)
#TODO unicode charactery sa mazu, lepsie je ich nahradit podobnymi