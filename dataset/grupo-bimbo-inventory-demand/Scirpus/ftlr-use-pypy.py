from csv import DictReader
from math import sqrt, log, expm1
from datetime import datetime

# TL; DR, the main training process starts on line: 250,
# you may want to start reading the code from there


##############################################################################
# parameters #################################################################
##############################################################################

# A, paths
train = '../input/train.csv'               # path to training file
test = '../input/test.csv'                 # path to testing file
submission = 'submission.csv'  # path of to be outputted submission file

# B, model
alpha = .02  # learning rate
beta = 1.   # smoothing parameter for adaptive learning rate
L1 = 0.     # L1 regularization, larger value means more regularized
L2 = 1.     # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 23             # number of weights to use
interaction = True     # whether to enable poly2 feature interactions

# D, training/validation
epoch = 3  # learn training data for N passes
holdout = 9  # use week holdout validation


##############################################################################
# class, function, generator definitions #####################################
##############################################################################

class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse regression with
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
            for i in range(L):
                for j in range(i+1, L):
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D
                    for k in range(j+1, L):
                        yield abs(hash(str(x[i]) + '_' + str(x[j]) +
                                  '_' + str(x[k]))) % D
                        for l in range(k+1, L):
                            yield abs(hash(str(x[i]) + '_' + str(x[j]) +
                                      '_' + str(x[k]) + '_' + str(x[l]))) % D

    def predict(self, x):
        ''' Get demand estimation on x

            INPUT:
                x: features

            OUTPUT:
                demand
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
            if ((L1 > 0) & (sign * z[i] <= L1)):
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # Raw Output
        return wTx

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: demand prediction of our model
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
            y: y: log(actual demand +1)
    '''

    for t, row in enumerate(DictReader(open(path))):
        ID = 0
        week = 0
        y = 0.
        if 'id' in row:
            ID = row['id']
            del row['id']
        if 'Demanda_uni_equil' in row:
            y = log(float(row['Demanda_uni_equil'])+1.)
            del row['Demanda_uni_equil']
        if 'Semana' in row:
            week = int(row['Semana'])
            del row['Semana']
        # build x
        x = []
        for key in row:
            if(key == 'Canal_ID' or key == 'Ruta_SAK' or
               key == 'Cliente_ID' or key == 'Producto_ID' or
               key == 'Agencia_ID'):
                value = row[key]
                # one-hot encode everything with hash trick
                index = abs(hash(key + '_' + value)) % D
                x.append(index)

        yield t, week, ID, x, y


##############################################################################
# start training #############################################################
##############################################################################
if __name__ == "__main__":
    print('Use PYPY!!!!')
    print('Remove the next line!!!!')
    exit(0)
    start = datetime.now()

    # initialize ourselves a learner
    learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)

    # start training
    for e in range(epoch):
        loss = 0.
        count = 0
        for t, week, ID, x, y in data(train, D):  # data is a generator
            #   t: just a instance counter
            #   week: you know what this is
            #   ID: id provided in original data
            #   x: features
            #   y: log(actual demand + 1)
            # step 1, get prediction from learner
            p = learner.predict(x)
            if((t % 100000) == 0):
                print(t)

            if ((holdout != 0) and (week >= holdout)):
                # step 2-1, calculate validation loss
                #           we do not train with the validation data so our
                #           validation loss is an accurate estimation
                #
                # holdout: train instances from day 1 to day N -1
                #            validate with instances from day N and after
                #
                loss += (max(0, p)-y)**2
                count += 1
            else:
                # step 2-2, update learner with demand information
                learner.update(x, p, y)

        count = max(count, 1)
        print('Epoch %d finished, validation RMSLE: %f, elapsed time: %s' %
              (e, sqrt(loss/count), str(datetime.now() - start)))

    #########################################################################
    # start testing, and build Kaggle's submission file #####################
    #########################################################################

    with open(submission, 'w') as outfile:
        outfile.write('id,Demanda_uni_equil\n')
        for t, date, ID, x, y in data(test, D):
            p = learner.predict(x)
            outfile.write('%s,%.3f\n' % (ID,
                                         expm1(max(0, p))))
            if((t % 100000) == 0):
                print(t)
    print('Finished')
