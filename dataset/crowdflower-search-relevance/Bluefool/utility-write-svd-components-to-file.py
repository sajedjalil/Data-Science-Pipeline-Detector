
from nltk.stem.porter import *
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

# Use Pandas to read in the training and test data
train = pd.read_csv("../input/train.csv").fillna("")
test  = pd.read_csv("../input/test.csv").fillna("")

# Print a sample of the training data
print(train.head())


sw=[]
s_data = []
s_labels = []
t_data = []
t_labels = []
s_all = []
t_all = []
s_id = []
t_id =[]
#stopwords tweak - more overhead
stop_words = ['http','www','img','border']
stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)
for stw in stop_words:
    sw.append("q"+stw)
    sw.append("z"+stw)
stop_words = text.ENGLISH_STOP_WORDS.union(sw)


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
            preprocessed_docs.append(final_doc)
        return preprocessed_docs


for i in range(len(train.id)):
    s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")])
    s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
    s_data.append(s)
    s_labels.append(str(train["median_relevance"][i]))
for i in range(len(test.id)):
    s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")])
    s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
    t_data.append(s)


tfv = TfidfVectorizer( min_df=3, ngram_range=(1, 2), max_features=None,strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',use_idf=1,smooth_idf=1,sublinear_tf=1,stop_words = 'english')

# Fit TFIDF
tfv.fit(s_data)
X =  tfv.transform(s_data)
X_test = tfv.transform(t_data)


svd = TruncatedSVD(n_components = 200,n_iter=5)
svd.fit(X)
s_data = svd.transform(X)
t_data = svd.transform(X_test)
s_data = pd.DataFrame(s_data)
t_data = pd.DataFrame(t_data)

s_data.to_csv("train_svd.csv", delimiter=",")
t_data.to_csv("test_svd.csv",  delimiter=",")


