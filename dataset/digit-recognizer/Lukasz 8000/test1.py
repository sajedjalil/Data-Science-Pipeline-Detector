# Fast LeNet5 CNN in Theano for GPU
# The code is based on the LeNet5 example: http://deeplearning.net/tutorial/lenet.html
# New features:
# 1. separate fit and predict methods
# 2. export/import of the trained model to a file
#
# Please note that here we run only 3 epochs, on a CPU within the 10 minutes time limit.
# We could run around 200 epochs on GPU in 10 minutes, obtaining a higher score.

import pandas as pd
import gzip
import os
import sys
import time
import numpy
import theano
import theano.tensor as Tpri
import sys, getopt
import time
import numpy
import numpy as np
import theano
import theano.tensor as T
from sklearn import preprocessing
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import pickle as cPickle

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        #self.x = input
        self.inp = input
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def pred_probs(self):
        pp = T.nnet.softmax(T.dot(self.inp, self.W) + self.b)
        return pp


def ReLU(x):
    y = T.maximum(0.0, x)
    return (y)


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(inumpyut,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inumpyut: theano.tensor.dmatrix
        :param inumpyut: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of inumpyut

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.inp = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]        
        

class CNN(object):

    def __getstate__(self):
        weights = [p.get_value() for p in self.params]
        #return (self.layer0.W, self.layer0.b, self.layer1.W, self.layer1.b, self.layer2.W,  
        #                       self.layer2.b, self.layer3.W, self.layer3.b)
        return weights

    def __setstate__(self, weights):
     #   (self.layer0.W, self.layer0.b, self.layer1.W, self.layer1.b, self.layer2.W, self.layer2.b, self.layer3.W, self.layer3.b) = state
        i = iter(weights)
        for p in self.params:
            p.set_value(i.__next__())
            


    def __init__(self, rng, input, nkerns, batch_size):
        # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (28, 28) is the size of MNIST images.
        self.layer0_input = input.reshape((batch_size, 1, 28, 28))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
        self.layer0 = LeNetConvPoolLayer(
            rng,
            input=self.layer0_input,
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5),
            poolsize=(2, 2)
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
        self.layer1 = LeNetConvPoolLayer(
            rng,
            input=self.layer0.output,
            image_shape=(batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2)
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        self.layer2_input = self.layer1.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        self.layer2 = HiddenLayer(
            rng,
            input=self.layer2_input,
            n_in=nkerns[1] * 4 * 4,
            n_out=500,
            activation=T.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        self.layer3 = LogisticRegression(input=self.layer2.output, n_in=500, n_out=10)

        # the cost we minimize during training is the NLL of the model
       # self.cost = self.layer3.negative_log_likelihood(y)

        self.errors = self.layer3.errors
        # create a list of all model parameters to be fit by gradient descent
        self.params = self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params
        
        
        

def fit(data, labels, filename = 'weights.pkl'):
    fit_predict(data, labels, filename = filename, action = 'fit') 

def predict(test_dataset, filename = 'weights.pkl' ):
    return fit_predict(data=[], labels=[], filename= filename, test_datasets=[test_dataset], action = 'predict')[0] 


def fit_predict(data, labels, action, filename, test_datasets = [], learning_rate=0.3, n_epochs=3, nkerns=[20, 50], batch_size=500, seed=8000):
    rng = numpy.random.RandomState(seed)
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    index = T.lscalar()  # index to a [mini]batch
    if action=='fit':
        NUM_TRAIN = len(data)
        if NUM_TRAIN % batch_size != 0: #if the last batch is not full, just don't use the remainder
            whole = (NUM_TRAIN // batch_size) * batch_size
            data = data[:whole]
            NUM_TRAIN = len(data) 

        # random permutation
        indices = rng.permutation(NUM_TRAIN)
        data, labels = data[indices, :], labels[indices]
        
        # batch_size == 500, splits (480, 20). We will use 96% of the data for training, and the rest to validate the NN while training
        is_train = numpy.array( ([0]* (batch_size - 20) + [1] * 20) * (NUM_TRAIN // batch_size))
        
        # now we split the dataset to test and valid datasets
        train_set_x, train_set_y = numpy.array(data[is_train==0]), labels[is_train==0]
        valid_set_x, valid_set_y = numpy.array(data[is_train==1]), labels[is_train==1]
        # compute number of minibatches 
        n_train_batches = len(train_set_y) // batch_size
        n_valid_batches = len(valid_set_y) // batch_size

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print ('... building the model')
        # allocate symbolic variables for the data
        epoch = T.scalar()
        #index = T.lscalar()  # index to a [mini]batch
        #x = T.matrix('x')  # the data is presented as rasterized images
        #y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
        
        # construct the CNN class
        classifier = CNN(
            rng=rng,
            input=x,
            nkerns = nkerns,
            batch_size = batch_size
        )

        train_set_x = theano.shared(numpy.asarray(train_set_x, dtype=theano.config.floatX))
        train_set_y = T.cast(theano.shared(numpy.asarray(train_set_y, dtype=theano.config.floatX)), 'int32')  
        valid_set_x = theano.shared(numpy.asarray(valid_set_x, dtype=theano.config.floatX)) 
        valid_set_y = T.cast(theano.shared(numpy.asarray(valid_set_y, dtype=theano.config.floatX)), 'int32')
        
        validate_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )

        cost = classifier.layer3.negative_log_likelihood(y)
        # create a list of gradients for all model parameters
        grads = T.grad(cost, classifier.params)

        # specify how to update the parameters of the model as a list of (variable, update expression) pairs
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(classifier.params, grads)
        ]

        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules defined in `updates`
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )




        ###############
        # TRAIN MODEL #
        ###############
        print ('... training')
        best_iter = 0
        test_score = 0.
        start_time = time.clock()
        epoch = 0

        # here is an example how to print the current value of a Theano variable: print test_set_x.shape.eval()
        
        # start training
        while (epoch < n_epochs):
            epoch = epoch + 1   
            for minibatch_index in range(n_train_batches):
                minibatch_avg_cost = train_model(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index
                if (epoch) % 5  == 0 and minibatch_index==0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

        ###############
        # PREDICTIONS #
        ###############

        # save and load
        f = open(filename, 'wb')
        cPickle.dump(classifier.__getstate__(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        end_time = time.clock()              
        #print >> sys.stderr, ('The code ran for %.2fm' % ((end_time - start_time) / 60.))


    if action == 'predict':
        # construct the CNN class
        classifier_2 = CNN(
            rng=rng,
            input=x,
            nkerns = nkerns,
            batch_size = batch_size
            )
        
        print ("....")


        f = open(filename, 'rb')
        classifier_2.__setstate__(cPickle.load(f))
        f.close()


        RET = []
        for it in range(len(test_datasets)):
            test_data = test_datasets[it]
            N = len(test_data)
            test_data = theano.shared(numpy.asarray(test_data, dtype=theano.config.floatX))
            # just zeroes
            test_labels = T.cast(theano.shared(numpy.asarray(numpy.zeros(batch_size), dtype=theano.config.floatX)), 'int32')
        
            ppm = theano.function([index], classifier_2.layer3.pred_probs(),
                givens={
                    x: test_data[index * batch_size: (index + 1) * batch_size],
                    y: test_labels
                }, on_unused_input='warn')

            # p : predictions, we need to take argmax, p is 3-dim: (# loop iterations x batch_size x 2)
            p = [ppm(ii) for ii in range( N // batch_size)]  
            #p_one = sum(p, [])
            #print p
            p = numpy.array(p).reshape((N, 10))
            print (p)
            p = numpy.argmax(p, axis=1)
            p = p.astype(int)
            RET.append(p)
        return RET
        
def run():
    # read the data, labels
    dtype = np.float32
    data = np.loadtxt("../input/train.csv", dtype=dtype,
                           delimiter=',', skiprows=1)
    test_data = np.loadtxt("../input/test.csv", dtype=dtype,
                          delimiter=',', skiprows=1)
    print (data.shape)

    labels = data[:,0]
    data = data[:, 1:]

   
    # DO argmax
    #labels = np.argmax(labels, axis=1)
        
    # normalization
    amean = np.mean(data)
    data = data - amean
    astd = np.std(data)
    data = data / astd
    # normalise using coefficients from training data
    test_data = (test_data - amean) / astd

    
    fit(data, labels)
    
  

    preds = predict(test_data)



    subm = np.empty((len(preds), 2))
    subm[:, 0] = np.arange(1, len(preds) + 1)
    subm[:, 1] = preds
    np.savetxt('submission.csv', subm, fmt='%d', delimiter=',',
               header='ImageId,Label', comments='')
   


if __name__ == '__main__':
    run()