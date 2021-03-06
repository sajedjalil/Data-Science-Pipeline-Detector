'''
Algorithm: Follow the regularized leader - proximal

Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf

This code is based on tinrtgu famous beat the benchmark with less than 1MB of memory.
In short,
this is an adaptive-learning-rate sparse logistic-regression with
efficient L1-L2-regularization

https://www.kaggle.com/c/criteo-display-ad-challenge/discussion/10322

And adapted for the competition from
https://github.com/swapniel99/criteo

How to run in your terminal?

First save the file, example: FTRL_microsoft.py

With python3:
	python3 FTRL_microsoft.py

	To run with python3 you may need to install Python wrapper for MurmurHash (MurmurHash3) with:

	sudo pip install mmh3

With pypy:
	You can run the code with pypy as it will give you a huge speed-up (0nly took 5 minuties). To use the script
	with pypy on Ubuntu, first type

	sudo apt-get install pypy

	Then you need to get pure python MurmurHash3 implementation to run with pypy from the
	follwoing link and then put the file to current directory

	https://raw.githubusercontent.com/wc-duck/pymmh3/master/pymmh3.py

Then use the pypy interpreter to run the script

pypy FTRL_microsoft.py
'''

from csv import DictReader
from datetime import datetime
import random
from math import exp, log, sqrt, copysign
from operator import xor

from itertools import combinations
import numpy as np
from numba import jit


# try to run pure python MurmurHash3 implementation with pypy
# download pure pymmh3 from https://raw.githubusercontent.com/wc-duck/pymmh3/master/pymmh3.py

#from pymmh3 import hash # uncomment this line if you run this code with pypy, and comment the next import

# ... otherwise try with a fast c-implementation to run with python3...
from mmh3 import hash # comment this line if you run this code with pypy, and uncomment the previous import


class DataGenerator(object):
    def __init__(self, interaction=2, signed=True):
        # feature related parameters
        self.signed = signed  # Use signed hash? Set to False for to reduce number of hash calls
        self.interaction = interaction  # whether to enable poly2 feature interactions

    def get_x(self, csv_row):
        """
        # Apply hash trick of the original csv row
        # for simplicity, we treat both integer and categorical features as categorical
        # INPUT:
        #     csv_row: a csv dictionary, ex: {'Lable': '1', 'I1': '357', 'I2': '', ...}
        #     D: the max index that we can hash to
        # OUTPUT:
        #     x: a list of indices that its value is 1
        """
        fullind = [hash(key + '=' + value.lower()) for key, value in csv_row.items()]
        # Creating interactions using XOR
        if self.interaction > 1:
            fullind_interact = fullind[:]

            # two-ways
            fullind.extend([xor(i, j) for i, j in combinations(fullind_interact, 2)])

            # three-ways
            if self.interaction > 2:
                fullind.extend([xor(xor(i, j), k) for i, j, k in combinations(fullind_interact, 3)])

        x = {k: 0 for k in fullind}
        if self.signed:
            for k, v in x.items():
                x[k] += 1 if hash(str(k)) & 1 else -1
        else:
            for k, v in x.items():
                x[k] += 1

        return x  # x contains indices of features that have a value as number of occurrences

    def get_data(self, file_path, id_col, label_col=None, exclude_cols=None):

        label_in_row = False
        remove_cols = list()

        for i, row in enumerate(DictReader(open(file_path)), 1):

            p_id = row[id_col]
            del row[id_col]

            if i == 1:
                label_in_row = label_col in row.keys()

                if exclude_cols:
                    remove_cols = [col for col in exclude_cols if col in row.keys()]

            if label_in_row:
                y = 1. if row[label_col] == '1' else 0.
                # del row[label_col]

            if remove_cols:
                for col in remove_cols:
                    del row[col]

            if not label_in_row:
                yield p_id, self.get_x(row)
            else:
                yield p_id, self.get_x(row), y

    def get_shuffle_data(self, file_path, id_col, label_col=None, exclude_cols=None, chunksize=10000):

        label_in_row = False
        remove_cols = list()
        cached_x = list()

        for i, row in enumerate(DictReader(open(file_path)), 1):

            p_id = row[id_col]
            del row[id_col]

            if i == 1:
                label_in_row = label_col in row.keys()

                if exclude_cols:
                    remove_cols = [col for col in exclude_cols if col in row.keys()]

            if label_in_row:
                y = 1. if row[label_col] == '1' else 0.

            if remove_cols:
                for col in remove_cols:
                    del row[col]

            # if not label_in_row:
            #     yield p_id, self.get_x(row)

            if not i % chunksize == 0:
                cached_x.append(tuple([p_id, self.get_x(row), y]))
                continue

            random.shuffle(cached_x)
            for x in cached_x:
                yield x

            cached_x = list()


class FTRLClassifier(object):

    def __init__(self, D=2**24, lambda1=0.001, lambda2=0.001, alpha=.1):

        self.e = None

        # feature related parameters
        self.D = D  # number of weights use for learning

        # model related parameters
        self.lambda1 = lambda1  # L1 regularization, larger value means more regularized
        self.lambda2 = lambda2  # L2 regularization, larger value means more regularized

        self.alpha = alpha  # learning rate for sgd optimization

        self.adapt = 1.  # Use adagrad, sets it as power of adaptive factor. >1 will amplify adaptive measure and vice versa
        # fudge = .4997  # Fudge factor (ratio of positive class)

        # initialize our model
        # self.w = np.zeros((D,))  # weights
        # self.g = np.ones((D,)) # * fudge  # sum of historical gradients
        self.bias = 0.
        self.bias_g = 0.

        self.w = [0.] * D  # weights
        self.g = None  # sum of historical gradients

        # log
        self.log_batch = 10000  # batch log info after every N rows

    @staticmethod
    def logloss(p, y):
        """
        # A. Bounded logloss
        # INPUT:
        #     p: our prediction
        #     y: real answer
        # OUTPUT
        #     logarithmic loss of p given y
        """
        p = max(min(p, 1. - 10e-17), 10e-17)  # The bounds
        loss = -log(p) if y == 1. else -log(1. - p)
        return loss

    @staticmethod
    def get_p(x, w, bias):
        """
        # C. Get probability estimation on x
        # INPUT:
        #     x: features
        #     w: weights
        # OUTPUT:
        #     probability of p(y = 1 | x; w)
        """
        wTx = sum([w[i] * xi for i, xi in x.items()]) + bias
        return 1. / (1. + exp(-max(min(wTx, 50.), -50.)))  # bounded sigmoid

    def update_w(self, x, p, y):
        """
        # D. Update given model
        # INPUT:
        #     w: weights
        #     n: a counter that counts the number of times we encounter a feature
        #        this is used for adaptive learning rate
        #     x: feature
        #     p: prediction of our model
        #     y: answer
        # OUTPUT:
        #     w: updated model
        #     n: updated count
        """
        diff = p - y

        # bias
        delta = diff * self.lambda1 * copysign(1., self.bias)
        if self.adapt > 0:
            self.bias_g += delta ** 2
            self.bias -= delta * self.alpha / (sqrt(self.bias_g) ** self.adapt)

        else:
            self.bias -= delta * self.alpha

        if self.adapt > 0:
            for i, x_i in x.items():
                w_i = self.w[i]

                delta = diff * x_i + self.lambda1 * copysign(1., w_i) + self.lambda2 * w_i

                g_i = self.g[i] + delta ** 2
                self.g[i] = g_i

                self.w[i] -= delta * self.alpha / (sqrt(g_i) ** self.adapt)  # Minimising log loss

        else:
            for i, x_i in x.items():
                w_i = self.w[i]

                delta = diff * x_i + self.lambda1 * copysign(1., w_i) + self.lambda2 * w_i
                self.w[i] -= delta * self.alpha

    def initialize(self, fudge, warm_start=False):
        self.e = 1

        if warm_start:
            self.w = [0.] * self.D  # weights

        self.g = [fudge] * self.D  # sum of historical gradients

    def train_one_epoch(self, gen_data, fudge=.5, adapt=None, warm_start=True):

        if not warm_start:
            print('re-initialized model')
            self.initialize(fudge, warm_start=False)

        if not self.g:  # no history
            self.initialize(fudge, warm_start=True)

        if adapt is not None:
            self.adapt = adapt
        else:
            self.adapt = 0.

        start_time = datetime.now()
        print('{}: train no {} epoch'.format(start_time, self.e))
        loss = 0.
        lossb = 0.
        for t, (p_id, row, y) in enumerate(gen_data):

            # main training procedure
            x = {k % self.D: v for k, v in row.items()}
            p = self.get_p(x, self.w, self.bias)

            # for progress validation, useless for learning our model
            lossb += self.logloss(p, y)

            self.update_w(x, p, y)  # step 3, update model with answer

            if t % self.log_batch == 0 and t > 1:
                loss += lossb
                print(
                    '{}\tepoch-> {}\tencountered: {}\tlogloss, current whole: {}\tcurrent batch: {}'.format(
                        datetime.now() - start_time, self.e, t, loss / t, lossb / self.log_batch))

                lossb = 0.

        print('{}\tfinish training {} epoch\n'.format(datetime.now() - start_time, self.e,))
        self.e += 1
        return loss / t

    def predict_file(self, gen_data, result_path):
        with open(result_path, 'w') as outfile:
            start_time = datetime.now()

            outfile.write('id,target\n')
            for t, (p_id, row) in enumerate(gen_data):

                x = {k % self.D: v for k, v in row.items()}
                p = self.get_p(x, self.w, self.bias)
                outfile.write('{},{}\n'.format(p_id, p))

                if t % self.log_batch == 0 and t > 1:
                    print('{}\ttest rows encountered: {}'.format(datetime.now() - start_time, t))

        print('{}\tfinish prediction\n\n'.format(datetime.now()))


def main():
    # dataset parameters #################################################################
    train = '../input/train.csv'  # path to training file
    test = '../input/test.csv'  # path to testing file
    submission = 'submission.csv'  # path of to be outputted submission file
    subm_template = 'subm_{task}_epoch{epoch:02d}_loss{loss:.3f}.csv'

    exclude_cols = list()
    label_col = 'target' # can't let the model peek the answer

    random.seed(42)
    rng = random.getstate()
    random.setstate(rng)

    max_iter = 2

    gen = DataGenerator(interaction=2, signed=True)

    #clf = FTRLClassifier(D=2**24, lambda1=0.001, lambda2=0.001, alpha=.1)
    clf = FTRLClassifier(D=2**24, lambda1=0.002, lambda2=0.001, alpha=.01)
    loss = clf.train_one_epoch(
        gen.get_data(train, id_col='id', label_col=label_col, exclude_cols=exclude_cols, ),
        fudge=.10049, adapt=None)

    # test
    submission = subm_template.format(task='test', epoch=1, loss=loss)
    clf.predict_file(
        gen.get_data(test, id_col='id', exclude_cols=exclude_cols,), submission)

    clf.predict_file(
        gen.get_data(test, id_col='id', exclude_cols=exclude_cols,), 'submission.csv')


if __name__ == '__main__':
    main()