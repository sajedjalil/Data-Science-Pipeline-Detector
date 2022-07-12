from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

import seaborn as sns

import numpy as np

import math

import pandas as pd

import string, os, random

from sklearn.feature_selection import chi2, SelectKBest

from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler

from sklearn.pipeline import FeatureUnion

punc = string.punctuation

from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from scipy import sparse

from itertools import combinations

from textblob import TextBlob 

from sklearn.utils import check_X_y 

from sklearn.utils.validation import check_is_fitted

import matplotlib.pyplot as plt

from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import LabelEncoder

import regex as re

from scipy.sparse import hstack

from sklearn.pipeline import make_union

from sklearn.metrics import accuracy_score

le = LabelEncoder()

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output



""" TO READ  !!!!!!!!! 



if someone can run this notebook with all observations et post a comment that will be great my computer don't have enough ressource for this, thanks" 

"""





""" --------------------------- importation---------------------------------- """



train = pd.read_csv('../input/train.csv')[:500]

test = pd.read_csv('../input/test.csv')[:500]

subm = pd.read_csv('../input/sample_submission.csv')[:500]



"""---------------------------------- get label for multiclassification task----------------------------------"""



lab = ['toxic', 'severe_toxic', 'obscene', 'threat',

       'insult', 'identity_hate']



for i in lab :

    train[i] = le.fit_transform(train[i])

    

"""-------------------------------- Create all function and class needed ------------------------------------"""

    

class Lemmatizer(BaseEstimator):

    def __init__(self):

        self.l = WordNetLemmatizer()       

    def fit(self, x, y=None):

        return self

    def transform(self, x):

        x = map(lambda r:  ' '.join([self.l.lemmatize(i.lower()) for i in r.split()]), x)

        x = np.array(list(x))

        return x



lemma = Lemmatizer()



#----------------------------------------------



def tokenize(s):

    pattern = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

    return pattern.sub(r' \1 ', s).split()



#----------------------------------------------



def get_polarity(text):

    try:

        pol = TextBlob(text).sentiment.polarity

    except:

        pol = 0.0

    return pol



#----------------------------------------------
"""


class SparseInteractions(BaseEstimator, TransformerMixin):

    def __init__(self, degree=2, feature_name_separator="_"):

        self.degree = degree

        self.feature_name_separator = feature_name_separator

    

    def fit(self, X, y=None):

        return self

        

    def transform(self, X):

        if not sparse.isspmatrix_csc(X):

            X = sparse.csc_matrix(X)

            

        if hasattr(X, "columns"):

            self.orig_col_names = X.columns

        else:

            self.orig_col_names = np.array([str(i) for i in range(X.shape[1])])

            

        spi = self._create_sparse_interactions(X)

        return spi



    def get_feature_names(self):

        return self.feature_names

    

    def _create_sparse_interactions(self, X):

        out_mat = []

        self.feature_names = self.orig_col_names.tolist()

        

        for sub_degree in range(2, self.degree + 1):

            for col_ixs in combinations(range(X.shape[1]), sub_degree):

                # add name for new column

                name = self.feature_name_separator.join(self.orig_col_names[list(col_ixs)])

                self.feature_names.append(name)

                

                # get column multiplications value

                out = X[:, col_ixs[0]]    

                for j in col_ixs[1:]:

                    out = out.multiply(X[:, j])



                out_mat.append(out)



        return sparse.hstack([X] + out_mat)

    """

#----------------------------------------------



def count_regexp_occ(regexp="", text=None):

    """ Simple way to get the number of occurence of a regex"""

    return len(re.findall(regexp, text))

    

"""---------------------------------- feature engineering---------------------------------- """



def get_indicators(train):

    train['word_count'] = train['comment_text'].apply(lambda x : len(x.split()))

    train['char_count'] = train['comment_text'].apply(lambda x : len(x.replace(" ","")))

    train['word_density'] = train['word_count'] / (train['char_count'] + 1)

    train['punc_count'] = train['comment_text'].apply(lambda x : len([a for a in x if a in punc]))

    train['total_length'] = train['comment_text'].apply(len)

    train['capitals'] = train['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))

    train['caps_vs_length'] = train.apply(lambda row: float(row['capitals'])/float(row['total_length']),

                                          axis=1)

    train['num_exclamation_marks'] =train['comment_text'].apply(lambda comment: comment.count('!'))

    train['num_question_marks'] = train['comment_text'].apply(lambda comment: comment.count('?'))

    train['num_punctuation'] = train['comment_text'].apply(

        lambda comment: sum(comment.count(w) for w in '.,;:'))

    train['num_symbols'] = train['comment_text'].apply(

        lambda comment: sum(comment.count(w) for w in '*&$%'))

    train['num_unique_words'] = train['comment_text'].apply(

        lambda comment: len(set(w for w in comment.split())))

    train['words_vs_unique'] = train['num_unique_words'] / train['word_count']

    train['polarity'] = train['comment_text'].apply(get_polarity)

    train["nb_fk"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"[Ff]\S{2}[Kk]", x))

    # Number of S word

    train["nb_sk"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"[Ss]\S{2}[Kk]", x))

    # Number of D words

    train["nb_dk"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"[dD]ick", x))

    # Number of occurence of You, insulting someone usually needs someone called : you

    train["nb_you"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))

    # Just to check you really refered to my mother ;-)

    train["nb_mother"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\Wmother\W", x))

    # Just checking for toxic 19th century vocabulary

    train["nb_ng"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\Wnigger\W", x))

    

for df in [train, test]:

    get_indicators(df)

    

    

"""---------------------------------- prepare vector and combine text feature and numeric features----------------------------------"""



num_features = [f_ for f_ in train.columns

                if f_ not in ["comment_text", "id"] + lab]



for f in num_features:

    all_cut = pd.cut(pd.concat([train[f], test[f]], axis=0), bins=20, labels=False, retbins=False)

    train[f] = all_cut.values[:train.shape[0]]

    test[f] = all_cut.values[train.shape[0]:]



train_num_features = train[num_features].values

test_num_features = test[num_features].values



train_text = train['comment_text'].fillna("")

test_text = test['comment_text'].fillna("")

all_text = pd.concat([train_text, test_text])





word_vectorizer = TfidfVectorizer(ngram_range =(1,3),

                             tokenizer=tokenize,

                             min_df=3, max_df=0.9,

                             strip_accents='unicode',

                             stop_words = 'english',

                             analyzer = 'word',

                             use_idf=1,

                             smooth_idf=1,

                             sublinear_tf=1)



char_vectorizer = TfidfVectorizer(ngram_range =(2,6),

                                 min_df=3, max_df=0.9,

                                 strip_accents='unicode',

                                 analyzer = 'char',

                                 stop_words = 'english',

                                 use_idf=1,

                                 smooth_idf=1,

                                 sublinear_tf=1,

                                 max_features=5000)



vectorizer = make_union(word_vectorizer, char_vectorizer)

vectorizer.fit(all_text)



train_text_matrix =vectorizer.transform(train['comment_text'])

test_text_matrix = vectorizer.transform(test['comment_text'])



train_features = hstack([train_text_matrix, train_num_features]).tocsr() 

test_features = hstack([test_text_matrix, test_num_features]).tocsr() 



"""----------------------------- Custom Algorithm from Jeremy notebook ---------------------------------------"""

# https://www.kaggle.com/jhoward/minimal-lstm-nb-svm-baseline-ensemble



class NbSvmClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1.0, dual=False, n_jobs=1):

        self.C = C

        self.dual = dual

        self.n_jobs = n_jobs

        self.classes_ = [0,1]



    def predict(self, x):

        # Verify that model has been fit

        check_is_fitted(self, ['_r', '_clf'])

        return self._clf.predict(x.multiply(self._r))



    def predict_proba(self, x):

        # Verify that model has been fit

        check_is_fitted(self, ['_r', '_clf'])

        return self._clf.predict_proba(x.multiply(self._r))



    def fit(self, x, y):

        # Check that X and y have correct shape

        x, y = check_X_y(x, y, accept_sparse=True)



        def pr(x, y_i, y):

            p = x[y==y_i].sum(0)

            return (p+1) / ((y==y_i).sum()+1)



        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))

        x_nb = x.multiply(self._r)

        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)

        return self



"""---------------------------- pipeline ----------------------------------------"""



pip = Pipeline([

    ('chi',SelectKBest(chi2,k=400)),

    ('multilabel', OneVsRestClassifier(NbSvmClassifier()))

])



"""--------------------------------classification ------------------------------------"""



for label in lab:

    print('... Processing {}'.format(label))

    y = train[label]

    classifier = pip

    classifier.fit(train_features, y)

    y_pred_X = classifier.predict(train_features)

    print('Training Accuracy is {}'.format(accuracy_score(y,y_pred_X)))

    test_y = classifier.predict(test_features)

    test_y_prob = classifier.predict_proba(test_features)[:,1]

    subm[label] = test_y_prob

    

"""----------------------------submission----------------------------------------"""

print('Mean Accuracy is {}'.format(np.mean((accuracy_score(y,y_pred_X)))))

#subm.to_csv('submission.csv', index=False)