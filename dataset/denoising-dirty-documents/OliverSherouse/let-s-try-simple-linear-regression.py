import concurrent.futures
import csv
import logging
import random

import numpy as np

import sklearn.base
import sklearn.ensemble
import sklearn.feature_extraction.image
import sklearn.preprocessing
import sklearn.cross_validation
import sklearn.linear_model
import sklearn.metrics
import sklearn.pipeline

import skimage.data

from pathlib import Path

from PIL import Image as image
from matplotlib import pyplot as plt
from sklearn.externals import joblib

TRAIN_DIR = Path("../input/train/")
TARGET_DIR = Path("../input/train_cleaned/")
TEST_DIR = Path("../input/test/")

PADDING = 1
CHUNKSIZE = 10000

logging.basicConfig(level=logging.DEBUG)

def predictor(est, X):
    logging.debug(X)
    return est.predict(X)
    
def chunk(X, chunksize=CHUNKSIZE):
    indices = list(range(0, len(X), chunksize))
    indices.append(len(X))
    for i in range(len(indices) - 1):
        yield X[indices[i]: indices[i + 1]]
    yield X[indices[-1]:]
        

class Detrender(sklearn.base.TransformerMixin):
    def __init__(self):
        super().__init__()
        self.model = None
        
    def fit(self, X, y):
        X = np.vstack(i.reshape(-1, 1) for i in X)
        y = np.hstack(i.flatten() for i in y)
        self.model = (sklearn.linear_model.LinearRegression().fit(X, y))
    
    def transform(self, X):
        predicted = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(predictor)(self.model, i.reshape(-1, 1))
            for i in X)
        return [i.reshape(j.shape) for i, j in zip(predicted, X)]
    
    def predict(self, X):
        return np.hstack(i.flatten() for i in self.transform(X))

def get_training_sets():
    X = list(joblib.Parallel(n_jobs=128, backend="threading")(
        joblib.delayed(skimage.data.imread)(str(i))
        for i in TRAIN_DIR.iterdir()))
    y = list(i / 255 for i in joblib.Parallel(n_jobs=128, backend="threading")(
        joblib.delayed(skimage.data.imread)(str(i))
        for i in TARGET_DIR.iterdir()))
    logging.info("Finished loading")
    return X, y

def get_random_forests(X, y):
    X = np.vstack(X)
    y = np.concatenate(y)
    assert(X.shape[0] == y.shape)
    model = sklearn.ensemble.RandomForestRegressor(n_jobs=-1,
        n_estimators=0, warm_start=True)
    indices = list(range(0, X.shape[0], CHUNKSIZE))
    indices.append(X.shape[0])
    for i in range(len(indices) - 1):
        if not (i + 1) % 5:
            logging.info("Fitting {} of {}".format(i + 1, len(indices) - 1))
        start, end = indices[i], indices[i + 1]
        model.set_params(n_estimators=model.get_params()["n_estimators"] + 1)
        model.fit(X[start: end], y[start: end])
    logging.info("Finished Training")
    return model


def get_index_and_array(path):
    imgarray = skimage.data.imread(str(path))
    index = []
    for i in range(imgarray.shape[0]):
        for j in range(imgarray.shape[1]):
            index.append("{}_{}_{}".format(path.stem, i + 1, j + 1))
    return index, imgarray

def get_test_set():
    test_set = tuple(joblib.Parallel(n_jobs=128, backend="threading")(
        joblib.delayed(get_index_and_array)(i)
        for i in TEST_DIR.iterdir()))
    logging.info("Finished Loading Test Set")
    return tuple(i[0] for i in test_set), list([i[1] for i in test_set])


def write_submission(index, predicted, path):
    with path.open('w', encoding='utf-8', newline='') as outf:
        writer = csv.writer(outf)
        writer.writerow(('id', 'value'))
        writer.writerows(zip(index, predicted))


def main():
    trainX, trainy = get_training_sets()
    pl = sklearn.pipeline.Pipeline(steps=(
        ("detrender", Detrender()),
    ))
    pl.fit(trainX, trainy)
    logging.info("Finished Training")
    index, testX = get_test_set()
    predicted = pl.predict(testX)
    
    logging.info("Finished Predicting")
    write_submission(index, predicted, Path("submission.csv"))
    image.fromarray(testX[0].astype('uint8'), 'L').save("raw.png")
    predicted_first = predicted[:testX[0].flatten().size].reshape(testX[0].shape)
    image.fromarray(predicted_first.astype('uint8'), 'L').save("clean.png")
    

if __name__ == "__main__":
    main()