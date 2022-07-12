#Classifying EEG signals with a convolutional neural network.

import logging
import os

import  numpy as np
import pandas as pd
from   multiprocessing import Pool
from    sklearn.metrics import roc_auc_score as auc
from     neon.datasets.dataset import Dataset
from    neon.backends import gen_backend
from   neon.experiments import FitExperiment as Fit
from  neon.layers import FCLayer, DataLayer, CostLayer, ConvLayer, PoolingLayer
from  neon.transforms import RectLin, Logistic, CrossEntropy
from   neon.models import MLP

logging.basicConfig(level=30)
logger = logging.getLogger()

# Train on the full dataset and generate a submission file if this flag
# is set to False. Otherwise, just validate on a subset.
validate = True


class GalData(Dataset):
    """
    Load the EEG data. In order to conserve memory, the minibatches
    are constructed on an on-demand basis. An instance of this class
    is created for each subject.
    """
    def __init__(self, **kwargs):
        self.nchannels = 32
        self.nclasses = 6
        self.__dict__.update(kwargs)
        self.loaded = False
        self.mean = None

    def setwin(self, **kwargs):
        self.__dict__.update(kwargs)
        assert self.winsize % self.subsample == 0
        # This many samples to be collected for a single observation.
        # The samples are picked by subsampling over a window, the size
        # of which is specified by winsize.
        self.nsamples = self.winsize // self.subsample

    def readfile(self, path, data, inds=None):
        df = pd.read_csv(path, index_col=0)
        filedata = np.float32(df.values)
        data = filedata if data is None else np.vstack((data, filedata))
        # Indices are saved to generate the submission file.
        inds = df.index if inds is None else np.hstack((inds, df.index))
        return data, inds

    def readfiles(self, dirname, serlist):
        """
        Read the serieses specified by argument.
        """
        basepath = os.path.join(os.pardir, 'input', dirname)
        data = labs = inds = None
        for series in serlist:
            filename = 'subj{}_series{}_data.csv'.format(self.subj, series)
            path = os.path.join(basepath, filename)
            data, inds = self.readfile(path, data, inds)
            if dirname == 'train':
                path = path.replace('data', 'events')
                labs, _ = self.readfile(path, labs)
            else:
                nrows = data.shape[0]
                labs = np.zeros((nrows, self.nclasses), dtype=np.float32)
        return data, labs, inds

    def prep(self, data):
        # TODO: Add your pre-processing code here.
        if self.mean is None:
            self.mean = data.mean()
            self.std = data.std()
        data -= self.mean
        data /= self.std
        return data

    def load(self, **kwargs):
        if self.loaded:
            return
        self.__dict__.update(kwargs)
        if validate:
            train, trainlabs, _ = self.readfiles('train', [7])
            test, testlabs, self.testinds = self.readfiles('train', [8])
        else:
            train, trainlabs, _ = self.readfiles('train', range(1, 9))
            test, testlabs, self.testinds = self.readfiles('test', [9, 10])
        self.inputs['train'] = self.prep(train)
        self.targets['train'] = trainlabs
        self.inputs['test'] = self.prep(test)
        self.targets['test'] = testlabs
        self.loaded = True

    def init_mini_batch_producer(self, batch_size, setname, predict):
        """
        This is called by neon once before training and then to switch
        from training to inference mode.
        """
        self.setname = setname
        # Number of elements in a single observation.
        obsize = self.nchannels * self.nsamples
        self.batchdata = np.empty((obsize, self.batch_size))
        self.batchtargets = np.empty((self.nclasses, self.batch_size))
        self.devdata = self.backend.empty(self.batchdata.shape)
        self.devtargets = self.backend.empty(self.batchtargets.shape)
        nrows = self.inputs[setname].shape[0]
        # We cannot use the first (winsize - 1) targets because there isn't
        # enough data before their occurrence.
        nbatches = (nrows - self.winsize + 1) // self.batch_size
        # This variable contains a mapping to pick the right target given
        # a zero-based index.
        self.inds = np.arange(nbatches * self.batch_size) + self.winsize - 1
        if predict is False:
            # Shuffle the map of indices if we are training.
            np.random.seed(0)
            np.random.shuffle(self.inds)
        return nbatches

    def get_mini_batch(self, batch):
        """
        Called by neon when it needs the next minibatch.
        """
        inputs = self.inputs[self.setname]
        targets = self.targets[self.setname]
        lag = self.winsize - self.subsample
        base = batch * self.batch_size
        for col in range(self.batch_size):
            # Use the saved mapping to retrieve the correct target.
            end = self.inds[base + col]
            self.batchtargets[:, col] = targets[end]
            # We back up from the index of the target and sample over
            # the defined window to construct an entire observation.
            rowdata = inputs[end-lag:end+1:self.subsample]
            # Transpose to make the data from each channel contiguous.
            self.batchdata[:, col] = rowdata.T.ravel()
        # Copy to the accelerator device (in case this is running on a GPU).
        self.devdata[:] = self.batchdata
        self.devtargets[:] = self.batchtargets
        return self.devdata, self.devtargets


class ConvNet(object):
    """
    The network definition.
    """
    def __init__(self, backend, dataset, subj):
        ad = {
            'type': 'adadelta',
            'lr_params': {'rho': 0.9, 'epsilon': 1e-10}
        }
        self.layers = []
        self.add(DataLayer(is_local=True, nofm=dataset.nchannels,
                           ofmshape=[1, dataset.nsamples]))
        self.add(ConvLayer(nofm=64, fshape=[1, 3],
                           activation=RectLin(), lrule_init=ad))
        self.add(PoolingLayer(op='max', fshape=[1, 2], stride=2))
        self.add(FCLayer(nout=128, activation=RectLin(), lrule_init=ad))
        self.add(FCLayer(nout=dataset.nclasses, activation=Logistic(),
                         lrule_init=ad))
        self.add(CostLayer(cost=CrossEntropy()))
        self.model = MLP(num_epochs=1, batch_size=128, layers=self.layers)
        self.backend = backend
        self.dataset = dataset

    def add(self, layer):
        self.layers.append(layer)

    def fit(self):
        Fit(model=self.model, backend=self.backend, dataset=self.dataset).run()
        return self

    def predict(self):
        ds = self.dataset
        outputs, targets = self.model.predict_fullset(self.dataset, 'test')
        predshape = (ds.inputs['test'].shape[0], ds.nclasses)
        preds = np.zeros(predshape, dtype=np.float32)
        labs = np.zeros_like(preds)
        # The output returned by the network is less than the number of
        # predictions to be made. We leave the missing predictions as zeros.
        start = ds.winsize - 1
        end = start + outputs.shape[1]
        preds[start:end] = outputs.asnumpyarray().T
        labs[start:end] = targets.asnumpyarray().T
        return labs, preds, ds.testinds


def run(subj):
    """
    Train and perform inference on data from a single subject.
    """
    try:
        backend = gen_backend(rng_seed=0, gpu='nervanagpu')
    except:
        backend = gen_backend(rng_seed=0)
    ds = GalData(subj=subj)
    sumpreds = None
    winlist = [1024] if validate else [768, 1024, 1280, 1536]
    for winsize in winlist:
        ds.setwin(winsize=winsize, subsample=16)
        network = ConvNet(backend, ds, subj)
        labs, preds, inds = network.fit().predict()
        if sumpreds is None:
            sumpreds = preds
        else:
            sumpreds += preds
    if validate:
        aucs = [auc(labs[:, i], sumpreds[:, i]) for i in range(ds.nclasses)]
        print('Subject %d AUC %.4f' % (subj, np.mean(aucs)))
    return labs, sumpreds, inds


if __name__ == '__main__':
    print('\'validate\' is %s' % validate)
    # Launch a separate process for each subject.
    nsubjects = 12
    pool = Pool()
    results = pool.map(run, range(1, nsubjects + 1))
    pool.close()
    labs = np.vstack([tup[0] for tup in results])
    preds = np.vstack([tup[1] for tup in results])
    if validate:
        # Compute AUC metric.
        nclasses = labs.shape[1]
        aucs = [auc(labs[:, i], preds[:, i]) for i in range(nclasses)]
        print('Mean AUC %.4f' % np.mean(aucs))
    else:
        # Generate submission file.
        columns = ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase',
                   'LiftOff', 'Replace', 'BothReleased']
        inds = np.hstack([tup[2] for tup in results])
        subm = pd.DataFrame(index=inds, columns=columns, data=preds)
        subm.to_csv('subm.csv', index_label='id', float_format='%.4f')
    print('Done.')