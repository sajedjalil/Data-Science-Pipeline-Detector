"""
Thanks to tinrtgu for the wonderful base script
Use pypy for faster computations.!

... further ripped from SRK -- https://www.kaggle.com/sudalairajkumar/ftrl-starter-with-leakage-vars

any ideas on how to use cython to further speed it up, would love to see it
"""
import csv
import time
from csv import DictReader
from math import exp, log, sqrt
from numba import jit

# TL; DR, the main training process starts on line: 250,
# you may want to start reading the code from there


##############################################################################
# parameters #################################################################
##############################################################################

# A, paths
data_path = "../input/"
train = data_path+'train.csv'               # path to training file
test = data_path+'test.csv'                 # path to testing file
submission = 'sub_proba.csv'  # path of to be outputted submission file

# B, model
alpha = .055  # learning rate
beta = 0.   # smoothing parameter for adaptive learning rate
L1 = 0.    # L1 regularization, larger value means more regularized
L2 = 0.     # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 26             # number of weights to use
interaction = False     # whether to enable poly2 feature interactions

# D, training/validation
epoch = 2       # learn training data for N passes
holdday = ''   # data after date N (exclusive) are used as validation

##############################################################################
# class, function, generator definitions #####################################
##############################################################################

class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index
        '''
        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D
        '''

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g
@jit
def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


def data(path, D):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''
    for t, row in enumerate(DictReader(open(path))):
        
        # process clicks
        y = 0.
        if 'is_attributed' in row:
            if row['is_attributed'] == '1':
                y = 1.
            del row['is_attributed'], row['attributed_time']
        
        try:
            click_id = row['click_id']
        except:
            click_id = ''
            
        # process id
        x = []
        
        # Parse hour and date
        date, time = row['click_time'].split(' ')
        hour = time.split(':')[0]
        x.append(abs(hash('hour_%s'%(hour))) % D)
        del row['click_time']
        
        # Add the rest of the features
        for k, v in row.items():
            x.append(abs(hash('%s_%s'%(k, v))) % D)
        
        # Add an interaction
        x.append(abs(hash('%s_app_chl_%s'%(row['channel'], row['app']))) % D)
        x.append(abs(hash('%s_os__chl_%s'%(row['channel'], row['os']))) % D)
        
        yield t, x, y, date, click_id



##############################################################################
# start training #############################################################
##############################################################################

start_time = time.time()

# initialize ourselves a learner
learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)

# start training
for e in range(epoch):
    loss = 0.
    count = 0.000001
    date = 0

    for t, x, y, date, _ in data(train, D):  # data is a generator
        #    t: just a instance counter
        # date: you know what this is
        #   ID: id provided in original data
        #    x: features
        #    y: label (click)

        # step 1, get prediction from learner
        p = learner.predict(x)

        if (holdday and date == holdday):
            # step 2-1, calculate validation loss
            #           we do not train with the validation data so that our
            #           validation loss is an accurate estimation
            #
            # holdafter: train instances from day 1 to day N
            #            validate with instances from day N + 1 and after
            #
            # holdout: validate with every N instance, train with others
            loss += logloss(p, y)
            count += 1.0
        else:
            # step 2-2, update learner with label (click) information
            learner.update(x, p, y)

        if t%1000000 == 0:
            print("Train Processed Rows : %sM ; %.5f ;%ss "%( int(t/1e+6), loss/count, '%0.0f'%(time.time()-start_time)))
        #if (time.time()-start_time )>3600: 
        #    break
       

##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################


start_time = time.time()

with open(submission, 'w') as outfile:
    outfile.write('click_id,is_attributed\n')
    for t, x, y, date, click_id in data(test, D):
        p = learner.predict(x)
        outfile.write('%s,%s\n' % (click_id, str(p)))
        if t%1000000 == 0:
            print("Test Processed Rows : %sM ; %ss "%( int(t/1e+6), '%0.0f'%(time.time()-start_time)))
