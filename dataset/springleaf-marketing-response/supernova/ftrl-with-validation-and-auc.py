#############################################################################################################
# modified by supernova
# auc calculator. Author: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py
# run it using pypy locally, about 3 times faster than python
#
# based on code below:
#classic tinrtgu's code
#https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/10927/beat-the-benchmark-with-less-than-1mb-of-memory
#modified by rcarson
#https://www.kaggle.com/jiweiliu
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
test='../input/train.csv'#'vali_100.tsv'
submission = 'ftrl_submission.csv'  # path of to be outputted submission file

# B, model
alpha = .005  # learning rate
beta = 1.   # smoothing parameter for adaptive learning rate
L1 = 1.E-9     # L1 regularization, larger value means more regularized
L2 = 1.     # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 24             # number of weights to use
interaction = False     # whether to enable poly2 feature interactions

# D, training/validation
epoch = 1     # learn training data for N passes
#holdafter = 9   # data after date N (exclusive) are used as validation
holdout = 10 # use every N training instance for holdout validation


##############################################################################
# class, function, generator definitions #####################################
##############################################################################

##############################################################################
# auc calculator. Author: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py
def tied_rank(x):
    """
    Computes the tied rank of elements in x.
    This function computes the tied rank of elements in x.
    Parameters
    ----------
    x : list of numbers, numpy array
    Returns
    -------
    score : list of numbers
            The tied rank f each element in x
    """
    sorted_x = sorted(zip(x,range(len(x))))
    r = [0 for k in x]
    cur_val = sorted_x[0][0]
    last_rank = 0
    for i in range(len(sorted_x)):
        if cur_val != sorted_x[i][0]:
            cur_val = sorted_x[i][0]
            for j in range(last_rank, i): 
                r[sorted_x[j][1]] = float(last_rank+1+i)/2.0
            last_rank = i
        if i==len(sorted_x)-1:
            for j in range(last_rank, i+1): 
                r[sorted_x[j][1]] = float(last_rank+i+2)/2.0
    return r

def auc(actual, posterior):
    """
    Computes the area under the receiver-operater characteristic (AUC)
    This function computes the AUC error metric for binary classification.
    Parameters
    ----------
    actual : list of binary numbers, numpy array
             The ground truth value
    posterior : same type as actual
                Defines a ranking on the binary numbers, from most likely to
                be positive to least likely to be positive.
    Returns
    -------
    score : double
            The mean squared error between actual and posterior
    """
    r = tied_rank(posterior)
    num_positive = len([0 for x in actual if x==1])
    num_negative = len(actual)-num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if actual[i]==1])
    auc = ((sum_positive - num_positive*(num_positive+1)/2.0) /
           (num_negative*num_positive))
    return auc
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
        self.z = [random() for k in range(D)]#[0.] * D
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
    i=0
    for t, row in enumerate(DictReader(open(path), delimiter=',')):
        # process id
        #print row
        try:
            ID=row['ID']
            del row['ID']
        except:
            pass
        # process clicks
        y = 0.
        target='target'#'IsClick' 
        if target in row:
            if float(row[target]) == 1:
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
print ('only used 1000 rows for train and test data, comment line 334 and line 350 locally')
start = datetime.now()
# initialize ourselves a learner
learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)
# start training
for e in range(epoch):
    loss_train = 0.
    loss_valid = 0.
    loss_valid_temp = 0.
    count = 0    
    count_train = 0.
    count_valid = 0.
    predlist=[]
    targetlist=[]
    for t,  x, y in data(train, D):  # data is a generator
        
        count+=1
        if count%holdout == 0:
            p = learner.predict(x)            
            loss_valid += logloss(p, y)
            loss_valid_temp = logloss(p, y)
            count_valid += 1     
            predlist.append(p)
            targetlist.append(y)       
            
        else:            
            p = learner.predict(x)
            loss_train += logloss(p, y)
            count_train += 1
            
            learner.update(x, p, y)
            
        if count%100==0:
            print ('time_used:%s\tepoch: %-4drows:%d\tt_logloss:%f\tv_logloss:%f\tv_auc:%f' %\
            (datetime.now() - start, e, count, loss_train/count_train, loss_valid/count_valid, \
            auc(targetlist, predlist)))  
            #t_logloss stands for current train_logloss, v for valid
        if count == 1000: break # comment this line locally
    
    count=0
    #import pickle
    #pickle.dump(learner,open('ftrlmodel_a-005_b-1_l1-E-9_l2-1_d-24_p-F_e-'+str(e)+'_ho-10.p','w'))
    print ('write result')
    ##############################################################################
    # start testing, and build Kaggle's submission file ##########################
    ##############################################################################
    with open(submission, 'w') as outfile:
        outfile.write('ID,target\n')
        for  ID, x, y in data(test, D):
            count+=1
            p = learner.predict(x)
            outfile.write('%s,%s\n' % (str(int(float(ID))), str(p)))
            if count%10000==0: print ('test rows:',count)
            if count == 1000: break # comment this line locally
            