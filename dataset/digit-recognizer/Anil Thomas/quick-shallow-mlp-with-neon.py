"""
Classifying MNIST digits with neon
To run this code locally, clone https://github.com/NervanaSystems/neon
"""

import logging
import sys
import numpy as np
from neon.backends import gen_backend
from neon.layers import FCLayer, DataLayer, CostLayer
from neon.models import MLP
from neon.transforms import RectLin, Logistic, CrossEntropy
from neon.datasets.dataset import Dataset
from neon.experiments import FitExperiment

logging.basicConfig(level=20)
logger = logging.getLogger()


class Mnist(Dataset):
    def load(self, **kwargs):
        dtype = np.float32
        train = np.loadtxt("../input/train.csv", dtype=dtype,
                           delimiter=',', skiprows=1)
        test = np.loadtxt("../input/test.csv", dtype=dtype,
                          delimiter=',', skiprows=1)
        # The first column contains labels.
        self.inputs['train'] = train[:, 1:] / 255.
        self.inputs['test'] = test / 255.
        self.targets['train'] = np.empty((train.shape[0], 10),
                                         dtype=dtype)
        self.targets['test'] = np.zeros((test.shape[0], 10),
                                        dtype=dtype)
        # Convert targets to one-hot encoding.
        for col in range(10):
            self.targets['train'][:, col] = train[:, 0] == col
        self.format()
        
        
class Network(object):
    def __init__(self, backend, dataset):
        layers = []
        layers.append(DataLayer(nout=784))
        layers.append(FCLayer(nout=1000, activation=RectLin()))
        layers.append(FCLayer(nout=10, activation=Logistic()))
        layers.append(CostLayer(cost=CrossEntropy()))
        self.model = MLP(num_epochs=10, batch_size=100, layers=layers)
        self.dataset = dataset
        
    def fit(self):
        self.experiment = FitExperiment(model=self.model, backend=backend,
                                        dataset=self.dataset)
        self.experiment.run()           
        
    def predict(self):
        outputs, targets = self.model.predict_fullset(self.dataset, 'test')
        preds = np.argmax(outputs.asnumpyarray().T, axis=1)
        return preds
        
        
if __name__ == '__main__':
    # If running locally on a GPU system, set command line parameter
    # to "cudanet". GPU backend can be installed from
    # https://github.com/NervanaSystems/cuda-convnet2
    gpu = sys.argv[1] if len(sys.argv) > 1 else None
    backend = gen_backend(rng_seed=0, gpu=gpu)
    dataset = Mnist()
    network = Network(backend, dataset)
    network.fit()
    preds = network.predict()
    subm = np.empty((len(preds), 2))
    subm[:, 0] = np.arange(1, len(preds) + 1)
    subm[:, 1] = preds
    np.savetxt('submission.csv', subm, fmt='%d', delimiter=',',
               header='ImageId,Label', comments='')
    logger.info('Done')
