import csv
# As is this script scores 0.71+ on the leaderboard. If you download and run
# at home, you can tweak the parameters as described in the Discussion
# to get 0.90+


import numpy as np
import scipy
import pandas
from sklearn.metrics import roc_auc_score
from numpy import fft
from numpy.random import randint
# Lasagne (& friends) imports
import theano
from nolearn.lasagne import BatchIterator, NeuralNet
from lasagne.objectives import aggregate, binary_crossentropy
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer,Conv1DLayer
from lasagne.updates import nesterov_momentum
from theano.tensor.nnet import sigmoid

# Silence some warnings from lasagne
import warnings
warnings.filterwarnings('ignore', '.*topo.*')
warnings.filterwarnings('ignore', module='.*lasagne.init.*')
warnings.filterwarnings('ignore', module='.*nolearn.lasagne.*')

SUBJECTS = list(range(1,13))
TRAIN_SERIES = list(range(1,9))
TEST_SERIES = [9,10]

N_ELECTRODES = 32
N_EVENTS = 6

# We train on TRAIN_SIZE randomly selected location each "epoch" (yes, that's
# not really an epoch). One-fifth of these locations are used for validation,
# hence the 5*X format, to make it clear what the number of validation points
# is.
TRAIN_SIZE = 5*1024



# We encapsulate the event / electrode data in a Source object. 

class Source:

    mean = None
    std = None

    def load_raw_data(self, subject, series):
        raw_data = [self.read_csv(self.path(subject, i, "data")) for i in series]
        self.data = np.concatenate(raw_data, axis=0)
        raw_events = [self.read_csv(self.path(subject, i, "events")) for i in series]
        self.events = np.concatenate(raw_events, axis=0)
    
    def normalize(self):
        self.data -= self.mean
        self.data /= self.std
        
    @staticmethod
    def path(subject, series, kind):
        prefix = "train" if (series in TRAIN_SERIES) else "test"
        return "../input/{0}/subj{1}_series{2}_{3}.csv".format(prefix, subject, series, kind)
    
    @staticmethod
    def read_csv(path):
        return pandas.read_csv(path, index_col=0).values
            
        
class TrainSource(Source):

    def __init__(self, subject, series_list):
        self.load_raw_data(subject, series_list)
        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0)
        self.normalize()
        self.principle_components = scipy.linalg.svd(self.data, full_matrices=False)
        self.std2 = self.data.std(axis=0)
        self.data /= self.std2

        
# Note that Test/Submit sources use the mean/std from the training data.
# This is both standard practice and avoids using future data in theano
# test set.
        
class TestSource(Source):

    def __init__(self, subject, series, train_source):
        self.load_raw_data(subject, series)
        self.mean = train_source.mean
        self.std = train_source.std
        self.principle_components = train_source.principle_components
        self.normalize()
        self.data /= train_source.std2
        

class SubmitSource(TestSource):

    def __init__(self, subject, a_series, train_source):
        TestSource.__init__(self, subject, [a_series], train_source)

    def load_raw_data(self, subject, series):
        [a_series] = series
        self.data = self.read_csv(self.path(subject, a_series, "data"))
        
        
# Lay out the Neural net.


class LayerFactory:
    """Helper class that makes laying out Lasagne layers more pleasant"""
    def __init__(self):
        self.layer_cnt = 0
        self.kwargs = {}
    def __call__(self, layer, layer_name=None, **kwargs):
        self.layer_cnt += 1
        name = layer_name or "layer{0}".format(self.layer_cnt)
        for k, v in kwargs.items():
            self.kwargs["{0}_{1}".format(name, k)] = v
        return (name, layer) 


SAMPLE_SIZE = 512 # Larger (2048 perhaps) would be better
DOWNSAMPLE = 8
TIME_POINTS = SAMPLE_SIZE // DOWNSAMPLE
    
class IndexBatchIterator(BatchIterator):
    """Generate BatchData from indices.
    
    Rather than passing the data into the fit function, instead we just pass in indices to
    the data.  The actual data is then grabbed from a Source object that is passed in at
    the creation of the IndexBatchIterator. Passing in a '-1' grabs a random value from
    the Source.
    
    As a result, an "epoch" here isn't a traditional epoch, which looks at all the
    time points. Instead a random subsamle of 0.8*TRAIN_SIZE points from the
    training data are used each "epoch" and 0.2 TRAIN_SIZE points are uses for
    validation.

    """
    def __init__(self, source, *args, **kwargs):
        super(IndexBatchIterator, self).__init__(*args, **kwargs)
        self.source = source
        if source is not None:
            # Tack on (SAMPLE_SIZE-1) copies of the first value so that it is easy to grab
            # SAMPLE_SIZE POINTS even from the first location.
            x = source.data
            self.augmented = np.zeros([len(x)+(SAMPLE_SIZE-1), N_ELECTRODES], dtype=np.float32)
            self.augmented[SAMPLE_SIZE-1:] = x
            self.augmented[:SAMPLE_SIZE-1] = x[0]
        self.Xbuf = np.zeros([self.batch_size, N_ELECTRODES, TIME_POINTS], np.float32) 
        self.Ybuf = np.zeros([self.batch_size, N_EVENTS], np.float32) 
    
    def transform(self, X_indices, y_indices):
        X_indices, y_indices = super(IndexBatchIterator, self).transform(X_indices, y_indices)
        [count] = X_indices.shape
        # Use preallocated space
        X = self.Xbuf[:count]
        Y = self.Ybuf[:count]
        for i, ndx in enumerate(X_indices):
            if ndx == -1:
                ndx = np.random.randint(len(self.source.events))
            sample = self.augmented[ndx:ndx+SAMPLE_SIZE]
            # Reverse so we get most recent point, otherwise downsampling drops the last
            # DOWNSAMPLE-1 points.
            X[i] = sample[::-1][::DOWNSAMPLE].transpose()
            if y_indices is not None:
                Y[i] = self.source.events[ndx]
        Y = None if (y_indices is None) else Y
        return X, Y
    

# Simple / Naive net. Borrows from Daniel Nouri's Facial Keypoint Detection Tutorial 
    
def create_net(train_source, test_source, batch_size=128, max_epochs=20): 
    
    batch_iter_train = IndexBatchIterator(train_source, batch_size=batch_size)
    batch_iter_test  = IndexBatchIterator(test_source, batch_size=batch_size)
    LF = LayerFactory()

    dense = 512 # larger (1024 perhaps) would be better
    
    layers = [
        LF(InputLayer, shape=(None, N_ELECTRODES, TIME_POINTS)), 
        LF(DropoutLayer, p=0.5),
        # This first layer condenses N_ELECTRODES down to num_filters.
        # Since the electrode results are reportedly highly reduntant this
        # should speed things up without sacrificing accuracy. It may
        # also increase stability. This was 8 in an earlier version.
        LF(Conv1DLayer, num_filters=4, filter_size=1),
        # Standard fully connected net from now on
        LF(DenseLayer, num_units=dense),
        LF(DropoutLayer, p=0.5),
        LF(DenseLayer, num_units=dense),
        LF(DropoutLayer, p=0.5),
        LF(DenseLayer, layer_name="output", num_units=N_EVENTS, nonlinearity=sigmoid)
    ]
    
    def loss(x,t):
        return aggregate(binary_crossentropy(x, t))
    
    
    nnet =  NeuralNet(
        y_tensor_type = theano.tensor.matrix,
        layers = layers,
        batch_iterator_train = batch_iter_train,
        batch_iterator_test = batch_iter_test,
        max_epochs=max_epochs,
        verbose=1,
        update = nesterov_momentum, 
        update_learning_rate = 0.02,
        update_momentum = 0.9,
        objective_loss_function = loss,
        regression = True,
        **LF.kwargs
        )

    return nnet


# Do the training.

train_indices = np.zeros([TRAIN_SIZE], dtype=int) - 1


def score(net, samples=256):
    """Compute the area under the curve, ROC score
    
    We take `samples` random samples and compute the ROC AUC
    score on those samples. 
    """
    source = net.batch_iterator_test.source
    test_indices = np.arange(len(source.events))
    np.random.seed(199)
    np.random.shuffle(test_indices)
    predicted = net.predict_proba(test_indices[:samples])
    actual = source.events[test_indices[:samples]]
    return roc_auc_score(actual.reshape(-1), predicted.reshape(-1))
    

def train(factory, subj, max_epochs=20, valid_series=[1,2]):
    # For better performance use the line below which uses all but
    # `valid_series` for training.
    tseries = sorted(set(TRAIN_SERIES) - set(valid_series))
    # tseries = sorted(set([3,4,7,8]) - set(valid_series))
    train_source = TrainSource(subj, tseries)
    test_source = TestSource(subj, valid_series, train_source)
    net = factory(train_source, test_source, max_epochs=max_epochs)
    net.fit(train_indices, train_indices)
    print("Score:", score(net))
    return (net, train_source)
 

def train_all(factory, max_epochs=40, valid_series=[1,2]):
    info = {}
    for subj in SUBJECTS:
        print("Subject:", subj)
        net, train_source = train(factory, subj, max_epochs, valid_series)
        info[subj] = (net.get_all_params_values(), train_source)
    return (factory, info)   
  
 
def make_submission(train_info, name):
    factory, info = train_info
    lines = ["id,HandStart,FirstDigitTouch,BothStartLoadPhase,LiftOff,Replace,BothReleased"]
    for subj in SUBJECTS:
        weights, train_source = info[subj]
        for series in [9,10]:
            print("Subject:", subj, ", series:", series)
            submit_source = SubmitSource(subj, series, train_source)  
            indices = np.arange(len(submit_source.data))
            net = factory(train_source=None, test_source=submit_source)
            net.load_weights_from(weights)
            probs = net.predict_proba(indices)
            for i, p in enumerate(probs):
                id = "subj{0}_series{1}_{2},".format(subj, series, i)
                lines.append(id + ",".join(str(x) for x in p))
    with open(name, 'w') as file:
        file.write("\n".join(lines))
        
        
if __name__ == "__main__":
    train_info = train_all(create_net, max_epochs=40) # Training for longer would likley be better
    make_submission(train_info, "naive_grasp.csv") 