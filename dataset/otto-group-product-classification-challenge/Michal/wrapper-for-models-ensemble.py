__author__ = 'Sushant'
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.optimize import minimize
from sklearn.metrics import log_loss
import numpy
from sklearn.cross_validation import StratifiedShuffleSplit

"""
Usage:

estimators = []
estimators.append(RandomForestClassifier(n_estimators = 100))
estimators.append(GMM(n_components = 9))

C_MC = MegaClassifier(estimators = estimators, xv_tries = 5)
C_MC. fit(X_train, y_train)

C_MC.predict_proba(X_test)

Description:

The MegaClassifier object automatically partitions training data in a 
stratified manner into 'xv_tries' number of folds (default 4), trains
all models in 'estimators' with the stratified training sets and records
their output on the stratified validation set.

During optimization it selects weights that result in minimization of 
averaged log-loss across all the validation sets.

"""

class StratifiedSplit(object):
    @staticmethod
    def train_test_split( X, y, test_size = 0.2):
        res = StratifiedShuffleSplit(y, n_iter=1, test_size=test_size)
        for ind_train, ind_test in res:
            X_train = []
            y_train = []
            X_test = []
            y_test = []
            for ind in ind_train:
                X_train.append(X[ind])
                y_train.append(y[ind])

            for ind in ind_test:
                X_test.append(X[ind])
                y_test.append(y[ind])

            return X_train, X_test, y_train, y_test


class MegaClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, xv_tries=4, test_size=0.2):
        self.estimators = estimators
        self.xv_tries = xv_tries
        self.test_size = test_size

    def fit(self, X, y):
        self.X_trains = []
        self.y_trains = []
        self.X_valids = []
        self.y_valids = []
        for i in xrange(self.xv_tries):
            Xt, Xv, yt, yv = StratifiedSplit.train_test_split(X, y, test_size=self.test_size)
            self.X_trains.append(Xt)
            self.X_valids.append(Xv)
            self.y_trains.append(yt)
            self.y_valids.append(yv)

        # train the classifiers
        self.all_xv_predictions = []

        for ind, Xt in enumerate(self.X_trains):
            cur_xv_predictions = []
            for estimator in self.estimators:
                #new_est = copy.deepcopy(estimator)
                #new_est.fit(Xt, self.y_trains[ind])
                estimator.fit(Xt, self.y_trains[ind])
                cur_xv_predictions.append(estimator.predict_proba(self.X_valids[ind]))
            self.all_xv_predictions.append(cur_xv_predictions)

        num_estimators = len(self.estimators)
        initial_weights = [1.0 / float(num_estimators) for i in xrange(num_estimators)]

        print ("Optimizing....")
        bounds = [(0, 1) for i in xrange(num_estimators)]
        constraints = {'type': 'eq', 'fun': lambda w: 1 - sum(w)}
        res = minimize(self.__find_best_blending_weights, initial_weights, bounds=bounds, constraints=constraints)
        self.final_weights = res.x
        print ("Optimization finished...")

        print ("Weights:")
        print (self.final_weights)

        for estimator in self.estimators:
            estimator.fit(X, y)


    def __find_best_blending_weights(self, weights):
        log_losses = []
        for ind1, xv_predictions in enumerate(self.all_xv_predictions):
            y_final_pred_prob = None
            for ind, est_predictions in enumerate(xv_predictions):
                if y_final_pred_prob is None:
                    y_final_pred_prob = weights[ind] * est_predictions
                else:
                    y_final_pred_prob = numpy.add(y_final_pred_prob, (weights[ind] * est_predictions))
            log_losses.append(log_loss(self.y_valids[ind1], y_final_pred_prob))

        log_losses = numpy.array(log_losses)
        return log_losses.mean()

    def predict_proba(self, X):

        y_final_pred_prob = None
        for ind, estimator in enumerate(self.estimators):
            y_pp_cur = estimator.predict_proba(X)
            if y_final_pred_prob is None:
                y_final_pred_prob = self.final_weights[ind] * y_pp_cur
            else:
                y_final_pred_prob = numpy.add(y_final_pred_prob, (self.final_weights[ind] * y_pp_cur))
        return y_final_pred_prob