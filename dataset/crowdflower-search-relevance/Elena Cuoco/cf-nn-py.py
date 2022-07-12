"""
@author: Elena Cuoco

Starting from
Beating the Benchmark
Search Results Relevance @ Kaggle
__author__ : Abhishek

Using Lasagne and nolearn
 


"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import pipeline
import theano
from lasagne import layers
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import InputLayer
from lasagne.nonlinearities import identity,sigmoid, tanh,rectify
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum,adagrad
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
import nltk
import os
from nltk.corpus import stopwords
import nltk.data
class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)
class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()



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
    
    #nltk.download()

    #STOPWORDS = nltk.corpus.stopwords.words('english')
    # Load the training file
    train = pd.read_csv('../input/train.csv').fillna(" ")
    test = pd.read_csv('../input/test.csv').fillna(" ")

    # we dont need ID columns
    idx = test.id.values.astype(int)
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)

    # create labels. drop useless columns
    y=  train['median_relevance'].values.astype(np.int32)

    train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

    # do some lambda magic on text columns
    traindata = list(train.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'],x['product_description']),axis=1))
    testdata = list(test.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'],x['product_description']),axis=1))


    tfv = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')


    # Fit TFIDF
    tfv.fit(traindata)
    X =  tfv.transform(traindata)
    X_test = tfv.transform(testdata)

    # Initialize SVD
    svd = TruncatedSVD(n_components=400)

    # Initialize the standard scaler
    scl = StandardScaler()


    # Create the pipeline
    preproc = pipeline.Pipeline([('svd', svd),('scl', scl) ])
    preproc.fit(X)
    X =  preproc.transform(X).astype(np.float32)
    X_test = preproc.transform(X_test).astype(np.float32)

    clf=clf= NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden1', layers.DenseLayer),
        ('dropout1', layers.DropoutLayer),
        ('hidden2', layers.DenseLayer),
        ('dropout2', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None,X.shape[1]),
    hidden1_num_units=512,  # number of units in hidden layer
    dropout1_p=0.5,
    hidden2_num_units=256,  # number of units in hidden layer
    hidden2_nonlinearity=rectify,
    dropout2_p=0.4,

    output_nonlinearity=softmax,  # output layer uses identity function
    output_num_units=5,  # target values

    # optimization method:
    update=adagrad,

    update_learning_rate=theano.shared(np.float32(0.1)),



    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.1, stop=0.0001),

        EarlyStopping(patience=10),
    ],
    use_label_encoder=False,

    batch_iterator_train=BatchIterator(batch_size=100),
    regression=False,  # flag to indicate we're dealing with regression problem
    max_epochs=100,  # we want to train this many epochs
    verbose=1,
    eval_size=0.1

    )
    X, y = shuffle(X, y, random_state=7)
    clf.fit(X, y)

    _, X_valid, _, y_valid = clf.train_test_split(X, y, clf.eval_size)




    y_pred=clf.predict(X_valid)
    score=quadratic_weighted_kappa(y_valid, y_pred)
    print("Best score: %0.3f" % score)



    preds = clf.predict(X_test)

    # Create your first submission file
    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission.to_csv("cf-sumbmission.csv", index=False)
