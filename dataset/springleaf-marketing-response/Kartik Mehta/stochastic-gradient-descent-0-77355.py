######################################80#######################################################################
## Logistic Regression using Stochastic Gradient Descent with adapted learning rate ##
## Currently regularization is not used
## Leaderboard score = 0.77355
## Takes 2-3 minutes to run on my laptop
#modified by kartik mehta
## https://www.kaggle.com/kartikmehtaiitd

# Giving due credit 
#classic tinrtgu's code
# https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/10927/beat-the-benchmark-with-less-than-1mb-of-memory
#modified by rcarson available as FTRL starter on kaggle code 
#https://www.kaggle.com/jiweiliu/springleaf-marketing-response/ftrl-starter-code
#############################################################################################################


from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random
import pickle


##############################################################################
# parameters #################################################################
##############################################################################

# A, paths
train='../input/train.csv'
test='../input/test.csv'
submission = 'sgd_subm.csv'  # path of to be outputted submission file

# B, model
alpha = .005  	# learning rate
beta = 1		
L1 = 0.     	# L1 regularization, larger value means more regularized
L2 = 0.     	# L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 24             # number of weights to use
interaction = False     # whether to enable poly2 feature interactions

# D, training/validation
epoch = 1       # learn training data for N passes
holdafter = None   # data after date N (exclusive) are used as validation
holdout = 100  # use every N training instance for holdout validation


##############################################################################
# class, function, generator definitions #####################################
##############################################################################

class gradient_descent(object):

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
        # G: squared sum of past gradients
        # w: weights
        self.w = [0.] * D  
        self.G = [0.] * D 

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

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # model
        w = self.w	

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            wTx += w[i]

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.G: increase by squared gradient
                self.w: weights
        '''
        # parameters
        alpha = self.alpha
        L1 = self.L1
        L2 = self.L2

        # model
        w = self.w
        G = self.G

        # gradient under logloss
        g = p - y
        # update z and n
        for i in self._indices(x):
            G[i] += g*g
#            w[i] -= alpha*1/sqrt(n[i]) * (g) ## Learning rate reducing as 1/sqrt(n_i) : ALso gives good performance but below code gives better results
            w[i] -= alpha/(beta+sqrt(G[i])) * (g) ## Learning rate reducing as alpha/(beta + sqrt of sum of g_i)

        self.w = w
        self.G = G

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

    for t, row in enumerate(DictReader(open(path), delimiter=',')):
      
        try:
            ID=row['ID']
            del row['ID']
        except:
            pass
        # process clicks
        y = 0.
        target='target'#'IsClick' 
        if target in row:
            if row[target] == '1':
                y = 1.
            del row[target]

        # extract date

        # turn hour really into hour, it was originally YYMMDDHH

        # build x
        x = []
        for key in row:
            value = row[key]

            # one-hot encode everything with hash trick
            index = abs(hash(key + '_' + value)) % D
            x.append(index)

        yield ID,  x, y


##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()

# initialize ourselves a learner
learner = gradient_descent(alpha, beta, L1, L2, D, interaction)

# start training
print('Training Learning started; total 150k training samples')
for e in range(epoch):
    loss = 0.
    count = 0
    for t,  x, y in data(train, D):  # data is a generator

        p = learner.predict(x)
        loss += logloss(p, y)
        learner.update(x, p, y)
        count+=1
        if count%15000==0:
            print('%s\tencountered: %d\tcurrent logloss: %f' % (datetime.now(), count, loss/count))

#import pickle
#pickle.dump(learner,open('sgd_adapted_learning.p','w'))

##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################
count=0
print('Testing started; total 150k test samples')
with open(submission, 'w') as outfile:
    outfile.write('ID,target\n')
    for  ID, x, y in data(test, D):
        count+=1
        if count%15000==0:
            print('%s\tencountered: %d' % (datetime.now(), count))
        p = learner.predict(x)
        outfile.write('%s,%s\n' % (ID, str(p)))