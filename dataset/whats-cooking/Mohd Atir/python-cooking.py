# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 00:08:19 2015

@author: Justfor
"""
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier


"""
Soft Voting/Majority Rule classifier

This module contains a Soft Voting/Majority Rule classifier for
classification clfs.

"""
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np


class EnsembleClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """ Soft Voting/Majority Rule classifier for unfitted clfs.

    Parameters
    ----------
    clfs : array-like, shape = [n_classifiers]
      A list of classifiers.
      Invoking the `fit` method on the `VotingClassifier` will fit clones
      of those original classifiers that will be stored in the class attribute
      `self.clfs_`.

    voting : str, {'hard', 'soft'} (default='hard')
      If 'hard', uses predicted class labels for majority rule voting.
      Else if 'soft', predicts the class label based on the argmax of
      the sums of the predicted probalities, which is recommended for
      an ensemble of well-calibrated classifiers.

    weights : array-like, shape = [n_classifiers], optional (default=`None`)
      Sequence of weights (`float` or `int`) to weight the occurances of
      predicted class labels (`hard` voting) or class probabilities
      before averaging (`soft` voting). Uses uniform weights if `None`.

    verbose : int, optional (default=0)
      Controls the verbosity of the building process.
        `verbose=0` (default): Prints nothing
        `verbose=1`: Prints the number & name of the clf being fitted
        `verbose=2`: Prints info about the parameters of the clf being fitted
        `verbose>2`: Changes `verbose` param of the underlying clf to
                     self.verbose - 2

    Attributes
    ----------
    classes_ : array-like, shape = [n_predictions]

    clf : array-like, shape = [n_predictions]
      The unmodified input classifiers

    clf_ : array-like, shape = [n_predictions]
      Fitted clones of the input classifiers

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from mlxtend.sklearn import EnsembleClassifier
    >>> clf1 = LogisticRegression(random_state=1)
    >>> clf2 = RandomForestClassifier(random_state=1)
    >>> clf3 = GaussianNB()
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eclf1 = EnsembleClassifier(clfs=[clf1, clf2, clf3],
    ... voting='hard', verbose=1)
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> eclf2 = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft')
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    >>> eclf3 = EnsembleClassifier(clfs=[clf1, clf2, clf3],
    ...                          voting='soft', weights=[2,1,1])
    >>> eclf3 = eclf3.fit(X, y)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>>
    """
    def __init__(self, clfs, voting='hard', weights=None, verbose=0):

        self.clfs = clfs
        self.named_clfs = {key: value for key, value in _name_estimators(clfs)}
        self.voting = voting
        self.weights = weights
        self.verbose = verbose

    def fit(self, X, y):
        """ Fit the clfs.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
        """
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % self.voting)

        if self.weights and len(self.weights) != len(self.clfs):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d clfs'
                             % (len(self.weights), len(self.clfs)))

        self.le_ = LabelEncoder()
        self.le_.fit(y)
        self.classes_ = self.le_.classes_
        self.clfs_ = [clone(clf) for clf in self.clfs]

        if self.verbose > 0:
            print("Fitting %d classifiers..." % (len(self.clfs)))

        for clf in self.clfs_:

            if self.verbose > 0:
                i = self.clfs_.index(clf) + 1
                print("Fitting clf%d: %s (%d/%d)" %
                      (i, _name_estimators((clf,))[0][0], i, len(self.clfs_)))

            if self.verbose > 2:
                if hasattr(clf, 'verbose'):
                    clf.set_params(verbose=self.verbose - 2)

            if self.verbose > 1:
                print(_name_estimators((clf,))[0][1])

            clf.fit(X, self.le_.transform(y))
        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """
        if self.voting == 'soft':

            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)

            maj = np.apply_along_axis(
                                      lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)

        maj = self.le_.inverse_transform(maj)
        return maj

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        avg = np.average(self._predict_probas(X), axis=0, weights=self.weights)
        return avg

    def transform(self, X):
        """ Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        If `voting='soft'`:
          array-like = [n_classifiers, n_samples, n_classes]
            Class probabilties calculated by each classifier.
        If `voting='hard'`:
          array-like = [n_classifiers, n_samples]
            Class labels predicted by each classifier.
        """
        if self.voting == 'soft':
            return self._predict_probas(X)
        else:
            return self._predict(X)

    def get_params(self, deep=True):
        """ Return estimator parameter names for GridSearch support"""
        if not deep:
            return super(EnsembleClassifier, self).get_params(deep=False)
        else:
            out = self.named_clfs.copy()
            for name, step in six.iteritems(self.named_clfs):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out

    def _predict(self, X):
        """ Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.clfs_]).T

    def _predict_probas(self, X):
        """ Collect results from clf.predict calls. """
        return np.asarray([clf.predict_proba(X) for clf in self.clfs_])
 
# A combination of Word lemmatization + LinearSVC model

traindf = pd.read_json("../input/train.json")

traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

testdf = pd.read_json("../input/test.json") 
testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       

corpustr = traindf['ingredients_string']
vectorizertr = TfidfVectorizer(stop_words='english', ngram_range = ( 1, 1),analyzer="word", 
                             max_df = .6 , binary=False , token_pattern=r'\w+' , sublinear_tf=False, norm = 'l2')
#vectorizertr = HashingVectorizer(stop_words='english',
#                             ngram_range = ( 1 , 2 ),analyzer="word", token_pattern=r'\w+' , n_features = 7000)
                             
tfidftr = vectorizertr.fit_transform(corpustr).todense()
corpusts = testdf['ingredients_string']
vectorizerts = TfidfVectorizer(stop_words='english', ngram_range = ( 1, 1),analyzer="word", 
                             max_df = .6 , binary=False , token_pattern=r'\w+' , sublinear_tf=False, norm = 'l2')

#vectorizerts = HashingVectorizer(stop_words='english',
#                             ngram_range = ( 1 , 2 ),analyzer="word", token_pattern=r'\w+' , n_features = 7000)

tfidfts = vectorizertr.transform(corpusts)

predictors_tr = tfidftr

targets_tr = traindf['cuisine']

predictors_ts = tfidfts


################################
# Initialize classifiers
################################

np.random.seed(1)
print("Ensemble: LR - linear SVC")
clf1 = LogisticRegression(random_state=1, C=7)
clf2 = LinearSVC(random_state=1, C=0.4, penalty="l2", dual=False)
nb = BernoulliNB()
rfc = RandomForestClassifier(random_state=1, criterion = 'gini', n_estimators=300)
sgd = SGDClassifier(random_state=1, alpha=0.00001, penalty='l2', n_iter=50)
    #'sgd__alpha': (0.00001, 0.000001),
    #'sgd__penalty': ('l2', 'elasticnet'),
    #'sgd__n_iter': (10, 50, 80),
#from sklearn.neighbors import KNeighborsClassifier
#nnn = KNeighborsClassifier(n_neighbors=15)

#eclf = EnsembleClassifier(clfs=[clf1, clf2,nb, rfc], weights=[3, 3, 1, 2])
#np.random.seed(1)
#for clf, label in zip([clf1, clf2, nb, rfc, eclf], 
#    ['Logistic Regression','linear-SVC','NB', 'RF', 'Ensemble']):
#    scores = cross_validation.cross_val_score(clf, predictors_tr,targets_tr, cv=3, scoring='accuracy')
#    print("Accuracy: %0.4f (+/- %0.5f) [%s]" % (scores.mean(), scores.std(), label))

eclf = EnsembleClassifier(clfs=[clf1, clf2,nb, rfc, sgd], weights=[2, 2, 1, 1,2])
np.random.seed(1)
for clf, label in zip([eclf], 
    ['Ensemble']):
    scores = cross_validation.cross_val_score(clf, predictors_tr,targets_tr, cv=2, scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.5f) [%s]" % (scores.mean(), scores.std(), label))


#eclf = EnsembleClassifier(clfs=[clf1, clf2, sgd], weights=[2,2,1])
#np.random.seed(1)
#for clf, label in zip([clf1, clf2, sgd, eclf], ['Logistic Regression','linear-SVC','SGD', 'Ensemble']):
#    scores = cross_validation.cross_val_score(clf, predictors_tr,targets_tr, cv=3, scoring='accuracy')
#    print("Accuracy: %0.4f (+/- %0.5f) [%s]" % (scores.mean(), scores.std(), label))

# LB 0.78922 with three clf; 0.7899 -> 0.79073, with four clf.

eclf2 = eclf.fit(predictors_tr,targets_tr)
predictions = eclf2.predict(predictors_ts)
testdf['cuisine'] = predictions
#testdf = testdf.sort('id' , ascending=True)
testdf = testdf.sort_values(by='id' , ascending=True)


##show the detail
# testdf[['id' , 'ingredients_clean_string' , 'cuisine' ]].to_csv("info_vote2.csv")

#for submit, no index
testdf[['id' , 'cuisine' ]].to_csv("just_python_cooking-vote1.csv", index=False)
