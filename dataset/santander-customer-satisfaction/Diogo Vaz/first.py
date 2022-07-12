# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#import numpy as np
import numpy.linalg as la

class SVMPredictor (object):
    def __init__(self,
                 kernel,
                 bias,
                 weights,
                 support_vectors,
                 support_vector_labels):
        self._kernel = kernel
        self._bias = bias
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels

    def predict(self, x):
        """
        Computes the SVM prediction on the given features x.
        """
        result = self._bias
        for z_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):
            result += z_i * y_i * self._kernel(x_i, x)
        return np.sign(result).item()
        
class Kernel(object):
    """Implements list of kernels from
    http://en.wikipedia.org/wiki/Support_vector_machine
    """
    @staticmethod
    def linear():
        def f(x, y):
            return np.inner(x, y)
        return f

    @staticmethod
    def gaussian(sigma):
        def f(x, y):
            exponent = -np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2))
            return np.exp(exponent)
        return f

    @staticmethod
    def _polykernel(dimension, offset):
        def f(x, y):
            return (offset + np.dot(x, y)) ** dimension
        return f

    @staticmethod
    def inhomogenous_polynomial(dimension):
        return Kernel._polykernel(dimension=dimension, offset=1.0)

    @staticmethod
    def homogenous_polynomial(dimension):
        return Kernel._polykernel(dimension=dimension, offset=0.0)

    @staticmethod
    def hyperbolic_tangent(kappa, c):
        def f(x, y):
            return np.tanh(kappa * np.dot(x, y) + c)
        return f

# Any results you write to the current directory are saved as output.
class SVMTrainer(object):
    def __init__ (self, kernel, c):
        self._kernel = kernel
        self._c = c
        def train(self, X, y):
            lagrange_multipliers = self._compute_multipliers(X, y)
            return self._construct_predictor(X, y, lagrange_multipliers)
            def _gram_matrix(self, X):
                n_samples, n_features = X.shape
                K = np.zeros((n_samples, n_samples))
                # TODO(tulloch) - vectorize
                for i, x_i in enumerate(X):
                    for j, x_j in enumerate(X):
                        K[i, j] = self._kernel(x_i, x_j)
                        return K
                        def _construct_predictor(self, X, y, lagrange_multipliers):
                            support_vector_indices = \
                            lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER
                            support_multipliers = lagrange_multipliers[support_vector_indices]
                            support_vectors = X[support_vector_indices]
                            support_vector_labels = y[support_vector_indices]
                            # http://www.cs.cmu.edu/~guestrin/Class/10701-S07/Slides/kernels.pdf
                            # bias = y_k - \sum z_i y_i  K(x_k, x_i)
                            # Thus we can just predict an example with bias of zero, and
                            # compute error.
                            bias = np.mean(
                                [y_k - SVMPredictor(
                                kernel=self._kernel,
                                bias=0.0,
                                weights=support_multipliers,
                                support_vectors=support_vectors,
                                support_vector_labels=support_vector_labels).predict(x_k)
                                for (y_k, x_k) in zip(support_vector_labels, support_vectors)])
                                #return SVMPredictor (
                                #    kernel=self._kernel,
                                #    bias=bias,
                                #    weights=support_multipliers,
                                #    support_vectors=support_vectors,
                                #    support_vector_labels=support_vector_labels)
def _compute_multipliers(self, X, y):
    n_samples, n_features = X.shape
    K = self._gram_matrix(X)
    # Solves
    # min 1/2 x^T P x + q^T x
    # s.t.
    #  Gx \coneleq h
    #  Ax = b
    P = cvxopt.matrix(np.outer(y, y) * K)
    q = cvxopt.matrix(-1 * np.ones(n_samples))
    # -a_i \leq 0
    # TODO(tulloch) - modify G, h so that we have a soft-margin classifier
    G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
    h_std = cvxopt.matrix(np.zeros(n_samples))
    # a_i \leq c
    G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
    h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)
    G = cvxopt.matrix(np.vstack((G_std, G_slack)))
    h = cvxopt.matrix(np.vstack((h_std, h_slack)))
    A = cvxopt.matrix(y, (1, n_samples))
    b = cvxopt.matrix(0.0)
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    # Lagrange multipliers
    return np.ravel(solution['x'])