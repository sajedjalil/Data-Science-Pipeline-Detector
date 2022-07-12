from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
import pandas as pd
import numpy as np
import time
import csv
##############################################################################
# parameters #################################################################
##############################################################################

# A, paths
path = '../input/'
train = '../input/clicks_train.csv'               # path to training file
test = '../input/clicks_test.csv'                 # path to testing file
submission = 'ftrl_output.csv'  # path of to be outputted submission file

# B, model
alpha = .1  # learning rate
beta = 0   # smoothing parameter for adaptive learning rate
L1 = 0     # L1 regularization, larger value means more regularized
L2 = 0     # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 20             # number of weights to use
interaction = False     # whether to enable poly2 feature interactions

# D, training/validation
epoch = 1       # learn training data for N passes
holdafter = None   # data after date N (exclusive) are used as validation
holdout = None  # use every N training instance for holdout validation


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

    for t, row in enumerate(DictReader(open(path))):
        # process id
        display_id = row['display_id']
        ad_id = row['ad_id']
        

        # process clicks
        y = 0.
        if 'clicked' in row:
            if row['clicked'] == '1':
                y = 1.
            del row['clicked']

        # build x
        x = []
        for key in row:
            value = row[key]

            # one-hot encode everything with hash trick
            index = abs(hash(key + '_' + value)) % D
            x.append(index)

        promoted_rows_byid = promoted_dict.get(ad_id,[])
        for idx,rows_val in enumerate(promoted_rows_byid):
            # one-hot encode
            index = abs(hash(promoted_header[idx] + '_' + rows_val)) % D
            x.append(index)
            
        display_rows_byid = events_dict.get(display_id,[])
        events_header = ['uuid','document_id','platform','country','state','number']
        for idx,rows_val in enumerate(display_rows_byid):
            # one-hot encode
            index = abs(hash(events_header[idx] + '_' + rows_val)) %D
            x.append(index)

        yield t,ad_id,display_id, x, y


##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()

# initialize ourselves a learner
learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)

################ Promoted Content ############
print('Reading Promoted Content File ...')
with open(path + 'promoted_content.csv') as content_file:
    content = csv.reader(content_file)
    promoted_header = next(content)[1:] ## document_id , campaign_id, advertiser_id
    print(promoted_header)
    promoted_dict = {}
    for idx,row_val in enumerate(content):
        promoted_dict[int(row_val[0])] = row_val[1:]
       # if idx == 1000000:
    #        break
    print('Promoted Content Length : ',len(promoted_dict))
del content
print('Finished....')
##############################################

#################### Event ###################
print('Reading Event File ....')
with open(path + 'events.csv') as event_file:
    events = csv.reader(event_file)
    #events_header = next(events)[1:]
    next(events) ### skip header
    events_dict = {}
    for idx,rows_val in enumerate(events):
       temp_list = rows_val[1:3] + list(rows_val[4])
       country_state_split = str(rows_val[5]).split('>')
       
       if len(country_state_split) ==3 :
           temp_list.extend(country_state_split[:])
       elif len(country_state_split) ==2:
           temp_list.extend(country_state_split[:] + ['Unknown'])
       elif len(country_state_split) ==1:
           temp_list.extend(country_state_split[:] + ['Unknown','Unknown'])
       else:
           temp_list.extend(['Unknown','Unknown','Unknown'])
       events_dict[int(rows_val[0])] = temp_list[:]

       #if idx % 10000 == 0:
    #       print('Proceed {}'.format(idx))
       if idx == 1000000:
           break
    print('Events Length : ',len(events_dict))
print('Finished...')
del events
################################################


# start training
print('Starting ....')
start_time = time.time()
for e in range(epoch):
    loss = 0.
    count = 0

    for t, ad_id,display_id, x, y in data(train, D):  # data is a generator
        #    t: just a instance counter
        #   ad_id: id provided in original data
        #   display_id : id provided in original data
        #    x: features
        #    y: label (click)

        # step 1, get prediction from learner
        p = learner.predict(x)

        if (holdafter and date > holdafter) or (holdout and t % holdout == 0):
            # step 2-1, calculate validation loss
            #           we do not train with the validation data so that our
            #           validation loss is an accurate estimation
            #
            # holdafter: train instances from day 1 to day N
            #            validate with instances from day N + 1 and after
            #
            # holdout: validate with every N instance, train with others
            loss += logloss(p, y)
            count += 1
        else:
            # step 2-2, update learner with label (click) information
            learner.update(x, p, y)

    #print('Epoch %d finished, validation logloss: %f, elapsed time: %s' % (
    #    e, loss/count, str(datetime.now() - start)))
        if t == 1000000:
            break
        #if t % 10000 ==0:
        #    print('Proceed {}'.format(t))
print('Finished ...')
print('Time is %f min' % ((time.time() - start_time) / 60))

#### Create Submission File ######
print('Writing File ....')
print('Predicting.....')
with open(submission, 'w') as outfile:
    outfile.write('display_id,ad_id,clicked_prob\n')
    for t, ad_id,display_id, x, y in data(test, D):
        p = learner.predict(x)
        #outfile.write('%s,%s\n' % (display_id, str(p)))
        outfile.write('%s,%s,%s\n' %(display_id,ad_id,str(p)))
        if t == 1000000:
            break
        #if t % 10000 == 0:
        #    print('Proceed {}'.format(t))
'''
test = pd.read_csv(test)      
output = pd.read_csv(submission)
test = test.merge(output,how='left')
test['clicked_prob'].fillna(test['clicked_prob'].mean(),inplace=True)
test = test.sort_values(['display_id','clicked_prob'],ascending=False)
test = test.groupby('display_id')['ad_id'].apply(lambda x:' '.join(map(str,x))).reset_index()
test.to_csv(submission,index=False)
print('Finished......')
'''
